"""人格自迭代：按周期用近期对话与长期记忆重写「当前使用」的那一份人格。

设计要点：
1. 只改当前生效的人格。目标人格按 AstrBot 自己的规则解析：
   会话强制人格 -> 会话所选人格 -> 全局默认人格；解析结果只写这一份，其余人格不动。
2. 取材两类：近期对话（AstrBot 自己的会话历史）、长期记忆（LivingMemory 只读）。
   任一来源缺失都只是少一段材料，不影响运行。
3. 写回前先备份，保留历史版本，支持回滚；字数上限按句子边界裁剪，硬性规则强制保留。
4. 定时器是插件内的 asyncio 循环，状态落在 plugin_data 下，重启不丢。
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register

PLUGIN_NAME = "astrbot_plugin_persona_evolve"
SYSTEM_DEFAULT_PERSONA_ID = "default"
TICK_SECONDS = 300
RETRY_SECONDS = 3600
MAX_RETRY = 3
SKIP_PERSONA_IDS = {"", "_chatui_default_"}

DIALOG_BUDGET = 4000
MEMORY_BUDGET = 1600


# --------------------------------------------------------------------------- 纯函数

_FENCE_HEAD_RE = re.compile(r"^```[^\n]*\n")
_FENCE_TAIL_RE = re.compile(r"\n```\s*$")
_LABEL_RE = re.compile(
    r"^(人格设定|人格提示词|人格|系统提示词|设定正文|System\s*Prompt|Persona)\s*[:：]\s*",
    re.IGNORECASE,
)


def clean_output(raw: str) -> str:
    """清掉模型爱加的代码块、标签和多余空行。"""
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = _FENCE_HEAD_RE.sub("", text)
        text = _FENCE_TAIL_RE.sub("", text)
    text = text.strip()
    text = text.strip('"').strip("“”").strip("「」")
    text = _LABEL_RE.sub("", text).strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_to(text: str, limit: int) -> str:
    """按句子边界裁剪，裁不动就硬切。"""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    head = text[:limit]
    floor = int(limit * 0.6)
    for sep in ("\n", "。", "！", "？", "；", ".", "!", "?", ";"):
        idx = head.rfind(sep)
        if idx >= floor:
            return head[: idx + 1].rstrip()
    return head.rstrip()


def fit_limits(text: str, max_chars: int, keep_lines: list[str]) -> str:
    """保留硬性规则，再把正文压到上限内；被裁掉的规则行补回开头。"""
    lines = list(dict.fromkeys(str(l).strip() for l in (keep_lines or []) if str(l).strip()))
    body = text.strip()
    missing = [l for l in lines if l not in body]
    if missing:
        prefix = "\n".join(missing)
        if len(prefix) + 1 >= max_chars:
            return prefix[:max_chars]
        body = truncate_to(body, max_chars - len(prefix) - 1)
        body = f"{prefix}\n{body}".strip()
    else:
        body = truncate_to(body, max_chars)

    dropped = [l for l in lines if l not in body]
    if dropped:
        prefix = "\n".join(dropped)
        budget = max_chars - len(prefix) - 1
        if budget <= 0:
            return prefix[:max_chars]
        body = f"{prefix}\n{truncate_to(body, budget)}".strip()
    return body


def _flatten_content(content: Any) -> str:
    """把 OpenAI 格式的 content（字符串 / 内容块数组）压成纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                kind = item.get("type")
                if kind in (None, "text"):
                    parts.append(str(item.get("text") or ""))
                elif kind in ("image_url", "image"):
                    parts.append("[图片]")
                elif item.get("text"):
                    parts.append(str(item.get("text") or ""))
        return " ".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        return str(content.get("text") or "").strip()
    return str(content).strip()


def _fmt_ts(ts: float) -> str:
    if not ts:
        return "从未"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _to_epoch(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        num = float(value)
        return num / 1000 if num > 1e11 else num
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip()).timestamp()
        except ValueError:
            return 0.0
    return 0.0


# --------------------------------------------------------------------------- 插件


@register(
    PLUGIN_NAME,
    "irislilymoon",
    "按周期用近期对话与长期记忆重写当前使用的人格，只改这一份",
    "1.0.0",
)
class PersonaEvolve(Star):
    """人格自迭代。"""

    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        self.data_root = Path(__file__).resolve().parents[2]
        self.data_dir = self.data_root / "plugin_data" / PLUGIN_NAME
        self._state: dict = {}
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._last_track_save = 0.0
        self._pcache: dict[tuple[str, str | None], tuple[float, str]] = {}
        self._register_apis()

    # ------------------------------------------------------------------ 生命周期

    async def initialize(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()
        now = time.time()
        if not self._state.get("first_seen_ts"):
            self._state["first_seen_ts"] = now
        if not self._state.get("baseline_ts"):
            interval = max(self._cfg_float("interval_days", 1.0), 1.0 / 24)
            if self._cfg_bool("run_on_start", False):
                self._state["baseline_ts"] = now - interval * 86400 - 1
            else:
                self._state["baseline_ts"] = now
        self._save_state()
        self._task = asyncio.create_task(self._loop(), name="persona_evolve_loop")
        logger.info(
            "[persona_evolve] 已启动，间隔 %s 天，上限 %s 字，按人格分别迭代（多活跃人格：%s）",
            self._cfg_float("interval_days", 1.0),
            self._cfg_int("max_chars", 800),
            "是" if self._cfg_bool("iterate_all_active", True) else "否",
        )

    async def terminate(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._save_state()

    # ------------------------------------------------------------------ 配置与状态

    def _cfg(self, key: str, default: Any = None) -> Any:
        try:
            value = self.config.get(key, default)
        except Exception:
            return default
        return default if value is None else value

    def _cfg_bool(self, key: str, default: bool = False) -> bool:
        return bool(self._cfg(key, default))

    def _cfg_int(self, key: str, default: int) -> int:
        try:
            return int(self._cfg(key, default))
        except (TypeError, ValueError):
            return default

    def _cfg_float(self, key: str, default: float) -> float:
        try:
            return float(self._cfg(key, default))
        except (TypeError, ValueError):
            return default

    def _keep_lines(self) -> list[str]:
        raw = self._cfg("keep_lines", [])
        if isinstance(raw, str):
            raw = [x for x in re.split(r"[\n,，]", raw)]
        try:
            return [str(x).strip() for x in raw if str(x).strip()]
        except TypeError:
            return []

    def _state_path(self) -> Path:
        return self.data_dir / "state.json"

    def _load_state(self) -> dict:
        try:
            data = json.loads(self._state_path().read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception as exc:
            logger.warning(f"[persona_evolve] 读取状态失败: {exc}")
            return {}

    def _save_state(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._state_path().write_text(
                json.dumps(self._state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(f"[persona_evolve] 保存状态失败: {exc}")

    def _debug(self, title: str, body: str) -> None:
        if not self._cfg_bool("debug", False):
            return
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            path = self.data_dir / "debug.log"
            if path.exists() and path.stat().st_size > 2 * 1024 * 1024:
                path.write_text("", encoding="utf-8")
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with path.open("a", encoding="utf-8") as fp:
                fp.write(f"\n===== {stamp} {title} =====\n{body}\n")
        except OSError:
            pass

    # ------------------------------------------------------------------ 定时

    async def _loop(self) -> None:
        await asyncio.sleep(15)
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - 兜底
                logger.error(f"[persona_evolve] 定时检查异常: {exc}", exc_info=True)
            await asyncio.sleep(TICK_SECONDS)

    async def _tick(self) -> None:
        if not self._cfg_bool("enabled", True):
            return
        now = time.time()
        try:
            targets = await self._select_due_targets(now)
        except Exception as exc:
            logger.error(f"[persona_evolve] 扫描待迭代人格失败: {exc}", exc_info=True)
            return
        if not targets:
            return

        for target, scope_entry in targets:
            pid = str(target.get("persona_id") or "")
            logger.info(f"[persona_evolve] 开始迭代人格 {pid}")
            result = await self._run_iteration(
                trigger="自动", target=target, scope=scope_entry
            )
            now = time.time()
            pstate = self._state.setdefault("personas", {}).setdefault(pid, {})
            pstate["last_attempt_ts"] = now
            if result.get("ok"):
                pstate["fail_count"] = 0
                pstate["last_run_ts"] = now
                pstate["last_run_iso"] = _fmt_ts(now)
                pstate["last_result"] = result.get("reason") or "ok"
                self._state["last_run_ts"] = now
                self._state["last_run_iso"] = _fmt_ts(now)
                self._state["last_result"] = result.get("reason") or "ok"
            elif result.get("skip_cycle"):
                pstate["fail_count"] = 0
                pstate["last_run_ts"] = now
                pstate["last_run_iso"] = _fmt_ts(now)
                pstate["last_result"] = result.get("reason") or "skipped"
            else:
                fails = int(pstate.get("fail_count") or 0) + 1
                pstate["fail_count"] = fails
                pstate["last_result"] = result.get("reason") or "failed"
                if fails >= MAX_RETRY:
                    pstate["last_run_ts"] = now
                    pstate["last_run_iso"] = _fmt_ts(now)
                    pstate["fail_count"] = 0
            self._save_state()
            if result.get("ok"):
                await self._notify(result, scope_entry)

    async def _select_due_targets(self, now: float) -> list[tuple[dict, dict]]:
        """列出这一轮该迭代的人格：各自独立计时，互不影响。"""
        interval = max(self._cfg_float("interval_days", 1.0), 1.0 / 24)
        scope = await self._scan_persona_scope()
        if not scope:
            return []
        pm = self.context.persona_manager
        personas_state: dict = self._state.setdefault("personas", {})
        baseline = float(self._state.get("baseline_ts") or 0) or now
        due: list[tuple[dict, dict]] = []

        for pid, entry in scope.items():
            if pid in SKIP_PERSONA_IDS:
                continue
            persona = pm.get_persona_v3_by_id(pid)
            if not persona:
                logger.debug(f"[persona_evolve] 人格 {pid} 已不在列表里，跳过")
                continue
            pstate = personas_state.setdefault(pid, {})
            last = float(pstate.get("last_run_ts") or 0) or baseline
            if now - last < interval * 86400:
                continue
            fails = int(pstate.get("fail_count") or 0)
            if fails > 0 and now - float(pstate.get("last_attempt_ts") or 0) < RETRY_SECONDS:
                continue
            latest = float(entry.get("latest") or 0)
            if latest and latest <= last:
                pstate["last_run_ts"] = now
                pstate["last_run_iso"] = _fmt_ts(now)
                pstate["last_result"] = "无新对话，跳过"
                self._save_state()
                continue
            due.append(
                (
                    {
                        "persona_id": pid,
                        "prompt": str(persona.get("prompt") or ""),
                        "source": "定时扫描",
                    },
                    entry,
                )
            )

        if not due:
            return []
        due.sort(key=lambda item: float(item[1].get("latest") or 0), reverse=True)
        if not self._cfg_bool("iterate_all_active", True):
            return due[:1]
        cap = max(self._cfg_int("max_personas_per_run", 3), 1)
        return due[:cap]

    async def _notify(self, result: dict, scope_entry: dict | None = None) -> None:
        if not self._cfg_bool("notify", False):
            return
        umo = self._notify_target(scope_entry)
        if not umo:
            return
        text = (
            f"人格已迭代：{result.get('persona_id')}，"
            f"{result.get('old_len')} → {result.get('new_len')} 字"
        )
        try:
            await self.context.send_message(umo, MessageChain().message(text))
        except Exception as exc:
            logger.debug(f"[persona_evolve] 通知发送失败: {exc}")

    def _notify_target(self, scope_entry: dict | None) -> str:
        """只在私聊 / webchat 里发通知，不往群里贴。"""
        def _private(umo: str) -> bool:
            low = umo.lower()
            return "friendmessage" in low or "webchat" in low

        for umo in list((scope_entry or {}).get("umos") or []):
            if _private(str(umo)):
                return str(umo)
        last = str((self._state.get("last_session") or {}).get("umo") or "")
        return last if last and _private(last) else ""

    # ------------------------------------------------------------------ 追踪当前人格

    @filter.on_llm_request()
    async def track_active_persona(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """记录最近一次真实请求用的是哪份人格，迭代时优先用它。"""
        try:
            umo = event.unified_msg_origin
            cm = self.context.conversation_manager
            conv_id = await cm.get_curr_conversation_id(umo)
            persona_id = None
            if conv_id:
                conv = await cm.get_conversation(umo, conv_id)
                if conv:
                    persona_id = conv.persona_id
            last = self._state.get("last_session") or {}
            changed = last.get("umo") != umo or last.get("persona_id") != persona_id
            now = time.time()
            if changed or now - self._last_track_save > 300:
                self._state["last_session"] = {
                    "umo": umo,
                    "persona_id": persona_id,
                    "ts": now,
                }
                self._last_track_save = now
                self._save_state()
        except Exception as exc:
            logger.debug(f"[persona_evolve] 记录会话失败: {exc}")

    # ------------------------------------------------------------------ 目标人格

    async def _default_persona_id(self) -> str:
        try:
            persona = await self.context.persona_manager.get_default_persona_v3()
            return str(persona.get("name") or SYSTEM_DEFAULT_PERSONA_ID)
        except Exception:
            return SYSTEM_DEFAULT_PERSONA_ID

    async def _session_persona_id(self, umo: str) -> str | None:
        try:
            cm = self.context.conversation_manager
            conv_id = await cm.get_curr_conversation_id(umo)
            if not conv_id:
                return None
            conv = await cm.get_conversation(umo, conv_id)
            return conv.persona_id if conv else None
        except Exception:
            return None

    async def _resolve_target(
        self, prefer_umo: str | None = None
    ) -> tuple[dict | None, str]:
        pm = self.context.persona_manager
        tries: list[tuple[str, str | None, str]] = []
        last = self._state.get("last_session") or {}
        if prefer_umo:
            tries.append(
                (prefer_umo, await self._session_persona_id(prefer_umo), "本次指令会话")
            )
        if last.get("umo"):
            tries.append((str(last["umo"]), last.get("persona_id"), "最近活跃会话"))

        for umo, conv_pid, source in tries:
            platform = umo.split(":", 1)[0] if ":" in umo else ""
            try:
                pid, persona, _forced, _webchat = await pm.resolve_selected_persona(
                    umo=umo,
                    conversation_persona_id=conv_pid,
                    platform_name=platform,
                )
            except Exception as exc:
                logger.debug(f"[persona_evolve] 解析人格失败 {umo}: {exc}")
                continue
            if pid and persona and pid not in SKIP_PERSONA_IDS:
                return {
                    "persona_id": str(pid),
                    "prompt": str(persona.get("prompt") or ""),
                    "source": source,
                }, ""

        try:
            convs, _total = await self.context.conversation_manager.get_filtered_conversations(
                page=1, page_size=30, include_history=False
            )
        except Exception as exc:
            logger.warning(f"[persona_evolve] 读取会话列表失败: {exc}")
            convs = []

        for conv in convs:
            platform = getattr(conv, "platform_id", "") or ""
            try:
                pid, persona, _forced, _webchat = await pm.resolve_selected_persona(
                    umo=conv.user_id,
                    conversation_persona_id=conv.persona_id,
                    platform_name=platform,
                )
            except Exception:
                continue
            if pid and persona and pid not in SKIP_PERSONA_IDS:
                return {
                    "persona_id": str(pid),
                    "prompt": str(persona.get("prompt") or ""),
                    "source": f"最近会话 {conv.user_id}",
                }, ""

        pid = await self._default_persona_id()
        persona = pm.get_persona_v3_by_id(pid)
        if persona:
            return {
                "persona_id": pid,
                "prompt": str(persona.get("prompt") or ""),
                "source": "全局默认人格",
            }, ""
        return None, "解析不到当前人格"

    async def _persona_exists(self, persona_id: str) -> bool:
        try:
            await self.context.persona_manager.get_persona(persona_id)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ 取材

    async def _conv_persona_id(
        self,
        umo: str,
        conv_persona_id: str | None,
        platform: str,
    ) -> str:
        """解析某个会话实际生效的人格：会话强制指定 > 会话所选 > 全局默认。

        不能只看 conversations.persona_id：WebUI 里给会话单独指定的人格存在
        preferences 的 session_service_config 里，列表字段是 None。
        私聊被指定成「私聊」人格就属于这种情况，漏掉会把私聊上下文算进默认人格。
        """
        key = (umo, conv_persona_id)
        hit = self._pcache.get(key)
        if hit and time.time() - hit[0] < 600:
            return hit[1]
        pid = ""
        try:
            resolved, persona, _forced, _webchat = (
                await self.context.persona_manager.resolve_selected_persona(
                    umo=umo,
                    conversation_persona_id=conv_persona_id,
                    platform_name=platform,
                )
            )
            if resolved and persona:
                pid = str(resolved)
        except Exception as exc:
            logger.debug(f"[persona_evolve] 解析 {umo} 的人格失败: {exc}")
        if not pid:
            pid = str(conv_persona_id or await self._default_persona_id())
        self._pcache[key] = (time.time(), pid)
        return pid

    async def _scan_persona_scope(self) -> dict[str, dict]:
        """扫最近窗口内的会话，按「实际生效的人格」分组。

        返回 {persona_id: {"convs": [(updated, conv), ...], "umos": [...], "latest": ts}}。
        umo 会同时用于按对话分割的长期记忆过滤，保证记忆边界与对话一致。
        """
        cm = self.context.conversation_manager
        days = max(self._cfg_float("recent_days", 7.0), 0.0)
        cutoff = time.time() - days * 86400 if days else 0.0
        try:
            convs, _total = await cm.get_filtered_conversations(
                page=1, page_size=60, include_history=False
            )
        except Exception as exc:
            logger.warning(f"[persona_evolve] 取会话失败: {exc}")
            return {}

        scope: dict[str, dict] = {}
        for conv in convs:
            updated = _to_epoch(getattr(conv, "updated_at", 0))
            if cutoff and updated and updated < cutoff:
                continue
            umo = str(getattr(conv, "user_id", "") or "")
            if not umo:
                continue
            pid = await self._conv_persona_id(
                umo,
                getattr(conv, "persona_id", None),
                str(getattr(conv, "platform_id", "") or ""),
            )
            entry = scope.setdefault(pid, {"convs": [], "umos": [], "latest": 0.0})
            entry["convs"].append((updated, conv))
            if umo not in entry["umos"]:
                entry["umos"].append(umo)
            entry["latest"] = max(entry["latest"], updated)
        for entry in scope.values():
            entry["convs"].sort(key=lambda item: item[0], reverse=True)
        return scope

    async def _collect_dialogs(
        self, convs: list[Any]
    ) -> list[str]:
        """把属于同一份人格的会话读成可读片段，只取最近的若干个。"""
        cm = self.context.conversation_manager
        max_convs = max(self._cfg_int("max_conversations", 3), 1)
        per_conv = max(self._cfg_int("messages_per_conversation", 24), 2)

        blocks: list[str] = []
        for conv in convs[:max_convs]:
            try:
                full = await cm.get_conversation(conv.user_id, conv.cid)
            except Exception as exc:
                logger.debug(f"[persona_evolve] 读取会话 {conv.cid} 失败: {exc}")
                continue
            if not full:
                continue
            lines = self._history_lines(getattr(full, "history", ""), per_conv)
            if not lines:
                continue
            title = (getattr(conv, "title", "") or conv.cid[:8]).strip()
            blocks.append(f"【会话：{title}】\n" + "\n".join(lines))
        return blocks

    @staticmethod
    def _history_lines(history_raw: Any, limit: int) -> list[str]:
        if isinstance(history_raw, str):
            try:
                records = json.loads(history_raw)
            except Exception:
                return []
        else:
            records = history_raw
        if not isinstance(records, list):
            return []
        out: list[str] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            role = record.get("role")
            if role not in ("user", "assistant"):
                continue
            text = _flatten_content(record.get("content"))
            if not text:
                continue
            text = re.sub(r"<system_reminder>.*?</system_reminder>", "", text, flags=re.S)
            text = re.sub(r"\s+", " ", text).strip()[:400]
            if not text:
                continue
            out.append(f"{'用户' if role == 'user' else '机器人'}：{text}")
        return out[-limit:] if limit > 0 else out

    def _read_memories(
        self, persona_id: str, session_ids: list[str] | None = None
    ) -> list[str]:
        """读长期记忆。

        LivingMemory 按对话分割记忆，atom 上有 session_id（就是 umo）。
        所以优先按「属于这个人格的那些会话」过滤，保证不会把私聊的记忆写进群聊人格；
        该人格还没有任何记忆时，再退回按 persona_id 过滤，最后才不加限制。
        """
        db_path = (
            self.data_root
            / "plugin_data"
            / "astrbot_plugin_livingmemory"
            / "livingmemory.db"
        )
        if not db_path.exists():
            return []
        days = max(self._cfg_float("recent_days", 7.0), 0.0)
        cutoff = time.time() - days * 86400 if days else 0.0
        limit = max(self._cfg_int("memory_limit", 20), 1)
        sessions = [str(s) for s in (session_ids or []) if str(s)][:40]

        attempts: list[tuple[str, tuple]] = []
        if sessions:
            marks = ",".join("?" * len(sessions))
            attempts.append((f"and session_id in ({marks})", tuple(sessions)))
        attempts.append(
            ("and (persona_id = ? or persona_id is null or persona_id = '')", (persona_id,))
        )
        attempts.append(("", ()))

        sql = (
            "select content from memory_atoms "
            "where status = 'active' and created_at >= ? {} "
            "order by importance desc, created_at desc limit ?"
        )
        rows: list[tuple] = []
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
            try:
                for cond, extra in attempts:
                    rows = conn.execute(
                        sql.format(cond), (cutoff, *extra, limit)
                    ).fetchall()
                    if rows:
                        break
            finally:
                conn.close()
        except Exception as exc:
            logger.warning(f"[persona_evolve] 读取长期记忆失败: {exc}")
            return []
        out: list[str] = []
        for (content,) in rows:
            text = str(content or "").strip()
            if text:
                out.append(text)
        return out

    # ------------------------------------------------------------------ 组装与调用

    def _build_system_prompt(self) -> str:
        max_chars = self._cfg_int("max_chars", 800)
        keep = self._keep_lines()
        keep_text = "；".join(keep) if keep else "（无）"
        return (
            "你是一个长期运行的角色机器人的「人格维护者」。"
            "你的工作是在保持角色连续性的前提下，依据真实发生过的对话与记忆，"
            "微调并重写这台机器人当前使用的人格设定。\n"
            "硬性要求：\n"
            "1. 只输出重写后的人格设定正文，不要解释、不要标题、不要代码块、不要引号包裹。\n"
            "2. 这是同一个角色的演化，不是新角色：保留原有名字、身份、称呼、关系、口癖、"
            "语言习惯与禁忌，不要改名或换设定。\n"
            "3. 保留原有的语气强度、表达尺度与行文结构，不要净化、不要说教、"
            "不要加入「作为 AI」「我只是程序」之类的元叙述。\n"
            "4. 可以依据素材细化：说话方式、偏好、与主人相处的方式、近期经历带来的态度变化。"
            "素材里没有的具体事实（人名、事件、时间、地点）一律不许编造。\n"
            "5. 素材与现有设定冲突时，以现有设定为准；素材只用于细化，不用于推翻角色。\n"
            "6. 不要写入「正在被改写」「这是人格文件」「迭代」之类的元信息。\n"
            f"7. 最终正文不超过 {max_chars} 个字符（含标点）。\n"
            f"8. 以下内容必须原样保留在正文中，可放在开头：{keep_text}\n"
            "9. 输出使用中文，沿用原文的排版习惯（换行分段、大写标记行等）。"
        )

    def _build_user_prompt(
        self,
        persona_id: str,
        current: str,
        dialogs: list[str],
        memories: list[str],
    ) -> str:
        days = self._cfg_float("recent_days", 7.0)
        extra = str(self._cfg("extra_prompt", "") or "").strip()
        parts: list[str] = []
        parts.append(
            f"【当前人格：{persona_id}，共 {len(current)} 字】\n"
            + (current or "（当前为空，需要从素材中建立）")
        )

        dialog_text = "\n\n".join(dialogs)
        if dialog_text:
            parts.append(f"【近期对话：最近 {days:g} 天】\n{truncate_to(dialog_text, DIALOG_BUDGET)}")
        else:
            parts.append("【近期对话】无")

        if memories:
            mem_text = "\n".join(f"- {m}" for m in memories)
            parts.append(f"【长期记忆摘录】\n{truncate_to(mem_text, MEMORY_BUDGET)}")
        else:
            parts.append("【长期记忆摘录】无")

        if extra:
            parts.append(f"【本轮附加要求】\n{extra}")
        parts.append("请输出重写后的人格设定正文。")
        return "\n\n".join(parts)

    async def _call_model(self, system_prompt: str, user_prompt: str, umo: str | None):
        provider_id = str(self._cfg("provider_id", "") or "").strip()
        if provider_id:
            provider = await self.context.provider_manager.get_provider_by_id(provider_id)
        else:
            provider = await self.context.get_using_provider_async(umo=umo)
        if provider is None:
            raise RuntimeError("没有可用的对话模型")
        timeout = max(self._cfg_int("timeout_seconds", 180), 30)
        response = await asyncio.wait_for(
            provider.text_chat(
                prompt=user_prompt,
                system_prompt=system_prompt,
                contexts=[],
            ),
            timeout=timeout,
        )
        model = ""
        try:
            model = str((provider.provider_config or {}).get("model") or "")
        except Exception:
            model = ""
        return str(getattr(response, "completion_text", "") or ""), model

    # ------------------------------------------------------------------ 版本

    def _version_dir(self, persona_id: str) -> Path:
        safe = re.sub(r"[^\w\u4e00-\u9fa5-]+", "_", persona_id)[:40] or "persona"
        return self.data_dir / "versions" / safe

    def _read_index(self, persona_id: str) -> list[dict]:
        path = self._version_dir(persona_id) / "index.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except FileNotFoundError:
            return []
        except Exception:
            return []

    def _write_index(self, persona_id: str, entries: list[dict]) -> None:
        path = self._version_dir(persona_id) / "index.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning(f"[persona_evolve] 写版本索引失败: {exc}")

    def _snapshot(
        self,
        persona_id: str,
        old_text: str,
        new_text: str,
        trigger: str,
        model: str,
    ) -> str:
        folder = self._version_dir(persona_id)
        folder.mkdir(parents=True, exist_ok=True)
        base = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        stamp = base
        seq = 1
        while (folder / f"{stamp}.old.txt").exists():
            seq += 1
            stamp = f"{base}-{seq}"
        (folder / f"{stamp}.old.txt").write_text(old_text, encoding="utf-8")
        (folder / f"{stamp}.new.txt").write_text(new_text, encoding="utf-8")
        entries = self._read_index(persona_id)
        entries.append(
            {
                "ts": stamp,
                "trigger": trigger,
                "old_len": len(old_text),
                "new_len": len(new_text),
                "model": model,
                "old_file": f"{stamp}.old.txt",
                "new_file": f"{stamp}.new.txt",
            }
        )
        keep = max(self._cfg_int("keep_versions", 20), 1)
        while len(entries) > keep:
            entries.pop(0)
        referenced = set()
        for entry in entries:
            referenced.add(str(entry.get("old_file") or ""))
            referenced.add(str(entry.get("new_file") or ""))
        for path in folder.glob("*.txt"):
            if path.name not in referenced:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        self._write_index(persona_id, entries)
        return stamp

    # ------------------------------------------------------------------ 主流程

    async def _run_iteration(
        self,
        trigger: str,
        dry_run: bool = False,
        prefer_umo: str | None = None,
        target: dict | None = None,
        scope: dict | None = None,
        force_write: bool = False,
    ) -> dict:
        result: dict = {
            "ok": False,
            "reason": "",
            "skip_cycle": False,
            "persona_id": "",
            "old_len": 0,
            "new_len": 0,
            "text": "",
            "dry_run": dry_run,
        }
        if self._lock.locked():
            result["reason"] = "已有一次迭代在进行中"
            return result
        async with self._lock:
            if target:
                persona_id = str(target.get("persona_id") or "")
                old_text = str(target.get("prompt") or "").strip()
                source = str(target.get("source") or "")
                scope_entry = scope or {}
            else:
                target, why = await self._resolve_target(prefer_umo)
                if not target:
                    result["reason"] = why or "解析不到当前人格"
                    result["skip_cycle"] = True
                    return result
                persona_id = target["persona_id"]
                old_text = (target["prompt"] or "").strip()
                source = target.get("source") or ""
                all_scope = await self._scan_persona_scope()
                scope_entry = all_scope.get(persona_id) or {}

            result["persona_id"] = persona_id
            result["old_len"] = len(old_text)

            if not await self._persona_exists(persona_id):
                result["reason"] = f"人格 {persona_id} 不在数据库里，跳过"
                result["skip_cycle"] = True
                return result

            scope_convs = [conv for _ts, conv in scope_entry.get("convs", [])]
            sessions: list[str] = list(scope_entry.get("umos", []))

            dialogs: list[str] = []
            if self._cfg_int("messages_per_conversation", 24) > 0:
                try:
                    dialogs = await self._collect_dialogs(scope_convs)
                except Exception as exc:
                    logger.warning(f"[persona_evolve] 取对话失败: {exc}")
            memories: list[str] = []
            if self._cfg_bool("use_livingmemory", True):
                try:
                    memories = await asyncio.to_thread(
                        self._read_memories, persona_id, sessions
                    )
                except Exception as exc:
                    logger.warning(f"[persona_evolve] 取记忆失败: {exc}")
            self._debug(
                "SCOPE",
                f"persona={persona_id}（来源：{source or '定时扫描'}）\n"
                f"会话数={len(scope_convs)} 记忆会话数={len(sessions)}\n"
                f"umos={json.dumps(sessions, ensure_ascii=False)}\n"
                f"对话片段={len(dialogs)} 记忆={len(memories)}",
            )
            extra = str(self._cfg("extra_prompt", "") or "").strip()

            if not (dialogs or memories or extra):
                result["reason"] = "没有可用素材（近期无对话、无记忆），本轮跳过"
                result["skip_cycle"] = True
                return result

            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(
                persona_id, old_text, dialogs, memories
            )
            self._debug("PROMPT", f"[system]\n{system_prompt}\n\n[user]\n{user_prompt}")

            umo = (self._state.get("last_session") or {}).get("umo") or prefer_umo
            try:
                raw, model = await self._call_model(system_prompt, user_prompt, umo)
            except asyncio.TimeoutError:
                result["reason"] = "模型响应超时"
                return result
            except Exception as exc:
                result["reason"] = f"调用模型失败: {exc}"
                return result
            self._debug("RAW_OUTPUT", raw)

            new_text = clean_output(raw)
            max_chars = max(self._cfg_int("max_chars", 800), 50)
            min_chars = max(self._cfg_int("min_chars", 200), 10)
            new_text = fit_limits(new_text, max_chars, self._keep_lines())

            if not new_text:
                result["reason"] = "模型输出为空"
                return result
            if len(new_text) < min_chars:
                result["reason"] = f"输出过短（{len(new_text)} 字 < 下限 {min_chars}），保留原人格"
                return result
            if new_text.strip() == old_text.strip():
                result["reason"] = "内容与当前一致，无需写入"
                result["skip_cycle"] = True
                return result

            result["text"] = new_text
            result["new_len"] = len(new_text)

            # 面板「立即迭代」传 force_write=True，不受全局 dry_run 影响
            want_dry = dry_run or (
                self._cfg_bool("dry_run", False) and not force_write
            )
            if want_dry:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                preview = self.data_dir / f"preview-{stamp}.txt"
                preview_name = ""
                try:
                    preview.write_text(
                        f"persona_id={persona_id}\n"
                        f"trigger={trigger}\nmodel={model}\n"
                        f"old_len={len(old_text)}\nnew_len={len(new_text)}\n"
                        f"---\n{new_text}\n",
                        encoding="utf-8",
                    )
                    preview_name = preview.name
                except OSError as exc:
                    logger.warning(f"[persona_evolve] 写预览失败: {exc}")
                result["ok"] = True
                result["dry_run"] = True
                result["preview"] = preview_name
                result["reason"] = (
                    f"演练完成，未写入人格（预览：{preview_name}）"
                    if preview_name
                    else "演练完成，未写入人格"
                )
                logger.info(f"[persona_evolve] {result['reason']}")
                return result

            stamp = self._snapshot(persona_id, old_text, new_text, trigger, model)
            try:
                await self.context.persona_manager.update_persona(
                    persona_id, system_prompt=new_text
                )
            except Exception as exc:
                result["reason"] = f"写回人格失败: {exc}"
                return result

            refreshed = self.context.persona_manager.get_persona_v3_by_id(persona_id)
            if refreshed and str(refreshed.get("prompt") or "").strip() != new_text.strip():
                result["reason"] = "写回后校验不一致，请检查"
                return result

            result["ok"] = True
            result["version"] = stamp
            result["reason"] = (
                f"人格 {persona_id} 已迭代：{len(old_text)} → {len(new_text)} 字（版本 {stamp}）"
            )
            logger.info(f"[persona_evolve] {result['reason']}")
            return result

    # ------------------------------------------------------------------ 面板 API

    def _register_apis(self) -> None:
        prefix = f"/{PLUGIN_NAME}/page"
        apis = [
            (f"{prefix}/state", self._api_state, ["GET"], "人格自迭代：面板状态"),
            (f"{prefix}/versions", self._api_versions, ["GET"], "人格自迭代：版本列表"),
            (f"{prefix}/version", self._api_version, ["GET"], "人格自迭代：读取版本"),
            (
                f"{prefix}/version/save",
                self._api_version_save,
                ["POST"],
                "人格自迭代：保存版本",
            ),
            (
                f"{prefix}/version/apply",
                self._api_version_apply,
                ["POST"],
                "人格自迭代：应用版本",
            ),
            (
                f"{prefix}/version/delete",
                self._api_version_delete,
                ["POST"],
                "人格自迭代：删除版本",
            ),
            (
                f"{prefix}/persona/save",
                self._api_persona_save,
                ["POST"],
                "人格自迭代：直接保存当前人格",
            ),
            (
                f"{prefix}/run",
                self._api_run,
                ["POST"],
                "人格自迭代：手动触发一次迭代",
            ),
        ]
        for route, handler, methods, desc in apis:
            try:
                self.context.register_web_api(route, handler, methods, desc)
            except Exception as exc:
                logger.warning(f"[persona_evolve] 注册面板接口 {route} 失败: {exc}")

    @staticmethod
    def _ok(data: Any) -> dict:
        return {"status": "ok", "data": data}

    @staticmethod
    def _fail(message: str) -> dict:
        return {"status": "error", "message": message}

    @staticmethod
    def _query_args() -> dict:
        try:
            from quart import request

            return {key: value for key, value in request.args.items()}
        except Exception:
            return {}

    @staticmethod
    async def _json_body() -> dict:
        try:
            from quart import request

            data = await request.get_json(silent=True)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    async def _check_persona(self, persona_id: str) -> bool:
        pid = str(persona_id or "").strip()
        if not pid or pid in SKIP_PERSONA_IDS:
            return False
        return await self._persona_exists(pid)

    def _find_entry(self, persona_id: str, ts: str) -> dict | None:
        for entry in self._read_index(persona_id):
            if str(entry.get("ts")) == str(ts):
                return entry
        return None

    def _persona_summary(self, persona_id: str, prompt: str, scope: dict) -> dict:
        entry = scope.get(persona_id) or {}
        pstate = (self._state.get("personas") or {}).get(persona_id) or {}
        return {
            "id": persona_id,
            "chars": len(str(prompt or "")),
            "in_use": bool(entry),
            "sessions": len(entry.get("umos") or []),
            "versions": len(self._read_index(persona_id)),
            "last_run": pstate.get("last_run_iso") or "",
            "last_result": pstate.get("last_result") or "",
        }

    async def _api_state(self):
        pm = self.context.persona_manager
        scope = await self._scan_persona_scope()
        personas: list[dict] = []
        seen: set[str] = set()
        for persona in list(getattr(pm, "personas", []) or []):
            pid = str(getattr(persona, "persona_id", "") or "")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            personas.append(
                self._persona_summary(pid, getattr(persona, "system_prompt", ""), scope)
            )
        for pid in scope:
            if pid in seen:
                continue
            seen.add(pid)
            persona = pm.get_persona_v3_by_id(pid) or {}
            personas.append(self._persona_summary(pid, persona.get("prompt", ""), scope))
        personas.sort(key=lambda item: (not item["in_use"], item["id"]))
        return self._ok(
            {
                "personas": personas,
                "interval_days": self._cfg_float("interval_days", 1.0),
                "max_chars": self._cfg_int("max_chars", 800),
                "enabled": self._cfg_bool("enabled", True),
                "dry_run": self._cfg_bool("dry_run", False),
            }
        )

    async def _api_versions(self):
        args = self._query_args()
        persona_id = str(args.get("persona") or "").strip()
        if not await self._check_persona(persona_id):
            return self._fail("人格不存在")
        persona = self.context.persona_manager.get_persona_v3_by_id(persona_id) or {}
        current = str(persona.get("prompt") or "")
        items = list(reversed(self._read_index(persona_id)))
        return self._ok(
            {
                "persona": persona_id,
                "current": current,
                "current_chars": len(current),
                "items": items,
            }
        )

    async def _api_version(self):
        args = self._query_args()
        persona_id = str(args.get("persona") or "").strip()
        ts = str(args.get("ts") or "").strip()
        kind = "new" if str(args.get("kind") or "").lower() == "new" else "old"
        if not await self._check_persona(persona_id):
            return self._fail("人格不存在")
        entry = self._find_entry(persona_id, ts)
        if not entry:
            return self._fail("版本不存在")
        file_path = self._version_dir(persona_id) / str(entry.get(f"{kind}_file") or "")
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            return self._fail(f"读取失败：{exc}")
        return self._ok(
            {"ts": ts, "kind": kind, "text": text, "chars": len(text), "entry": entry}
        )

    async def _api_version_save(self):
        body = await self._json_body()
        persona_id = str(body.get("persona") or "").strip()
        ts = str(body.get("ts") or "").strip()
        kind = "new" if str(body.get("kind") or "").lower() == "new" else "old"
        text = str(body.get("text") or "")
        if not await self._check_persona(persona_id):
            return self._fail("人格不存在")
        if not text.strip():
            return self._fail("内容不能为空")
        entries = self._read_index(persona_id)
        target = None
        for entry in entries:
            if str(entry.get("ts")) == ts:
                target = entry
                break
        if not target:
            return self._fail("版本不存在")
        file_path = self._version_dir(persona_id) / str(target.get(f"{kind}_file") or "")
        try:
            file_path.write_text(text, encoding="utf-8")
        except OSError as exc:
            return self._fail(f"写入失败：{exc}")
        target[f"{kind}_len"] = len(text)
        self._write_index(persona_id, entries)
        return self._ok({"ts": ts, "kind": kind, "chars": len(text)})

    async def _api_version_apply(self):
        body = await self._json_body()
        persona_id = str(body.get("persona") or "").strip()
        ts = str(body.get("ts") or "").strip()
        kind = "new" if str(body.get("kind") or "").lower() == "new" else "old"
        if not await self._check_persona(persona_id):
            return self._fail("人格不存在")
        entry = self._find_entry(persona_id, ts)
        if not entry:
            return self._fail("版本不存在")
        file_path = self._version_dir(persona_id) / str(entry.get(f"{kind}_file") or "")
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            return self._fail(f"读取失败：{exc}")
        if not text.strip():
            return self._fail("版本内容为空，不能应用")
        persona = self.context.persona_manager.get_persona_v3_by_id(persona_id) or {}
        current = str(persona.get("prompt") or "")
        self._snapshot(persona_id, current, text, f"面板应用-{ts}-{kind}", "panel")
        try:
            await self.context.persona_manager.update_persona(
                persona_id, system_prompt=text
            )
        except Exception as exc:
            return self._fail(f"写入人格失败：{exc}")
        logger.info(
            f"[persona_evolve] 面板应用版本 {ts}({kind}) 到 {persona_id}："
            f"{len(current)} → {len(text)} 字"
        )
        return self._ok({"chars": len(text), "old_chars": len(current)})

    async def _api_version_delete(self):
        body = await self._json_body()
        persona_id = str(body.get("persona") or "").strip()
        ts = str(body.get("ts") or "").strip()
        if not await self._check_persona(persona_id):
            return self._fail("人格不存在")
        entries = self._read_index(persona_id)
        kept = [entry for entry in entries if str(entry.get("ts")) != ts]
        if len(kept) == len(entries):
            return self._fail("版本不存在")
        folder = self._version_dir(persona_id)
        for entry in entries:
            if str(entry.get("ts")) != ts:
                continue
            for key in ("old_file", "new_file"):
                try:
                    (folder / str(entry.get(key) or "")).unlink(missing_ok=True)
                except OSError:
                    pass
        self._write_index(persona_id, kept)
        return self._ok({"ts": ts, "remaining": len(kept)})

    async def _api_persona_save(self):
        """直接改当前生效的人格，改前先存一份快照。"""
        body = await self._json_body()
        persona_id = str(body.get("persona") or "").strip()
        text = str(body.get("text") or "")
        if not await self._check_persona(persona_id):
            return self._fail("人格不存在")
        if not text.strip():
            return self._fail("内容不能为空")
        persona = self.context.persona_manager.get_persona_v3_by_id(persona_id) or {}
        current = str(persona.get("prompt") or "")
        if text.strip() == current.strip():
            return self._ok({"chars": len(text), "unchanged": True})
        label = str(body.get("trigger") or "").strip()
        if label not in ("面板手改", "演练确认"):
            label = "面板手改"
        stamp = self._snapshot(persona_id, current, text, label, "panel")
        try:
            await self.context.persona_manager.update_persona(
                persona_id, system_prompt=text
            )
        except Exception as exc:
            return self._fail(f"写入人格失败：{exc}")
        logger.info(
            f"[persona_evolve] 面板直接修改 {persona_id}："
            f"{len(current)} → {len(text)} 字（版本 {stamp}）"
        )
        return self._ok(
            {"chars": len(text), "old_chars": len(current), "version": stamp}
        )

    async def _api_run(self):
        """面板上手动跑一次：dry_run=true 只生成，false 直接写入。"""
        body = await self._json_body()
        persona_id = str(body.get("persona") or "").strip()
        dry = bool(body.get("dry_run"))
        if not await self._check_persona(persona_id):
            return self._fail("人格不存在")
        persona = self.context.persona_manager.get_persona_v3_by_id(persona_id) or {}
        target = {
            "persona_id": persona_id,
            "prompt": str(persona.get("prompt") or ""),
            "source": "面板手动触发",
        }
        try:
            scope = (await self._scan_persona_scope()).get(persona_id) or {}
        except Exception as exc:
            logger.warning(f"[persona_evolve] 面板触发时取会话失败: {exc}")
            scope = {}
        result = await self._run_iteration(
            trigger="面板演练" if dry else "面板手动",
            dry_run=dry,
            target=target,
            scope=scope,
            force_write=not dry,
        )
        if result.get("ok") and not result.get("dry_run"):
            now = time.time()
            pstate = self._state.setdefault("personas", {}).setdefault(persona_id, {})
            pstate["last_run_ts"] = now
            pstate["last_run_iso"] = _fmt_ts(now)
            pstate["last_result"] = result.get("reason") or ""
            pstate["fail_count"] = 0
            self._state["last_run_ts"] = now
            self._state["last_run_iso"] = _fmt_ts(now)
            self._state["last_result"] = result.get("reason") or ""
            self._save_state()
        return self._ok(
            {
                "ok": bool(result.get("ok")),
                "reason": result.get("reason") or "",
                "dry_run": bool(result.get("dry_run")),
                "persona": persona_id,
                "old_len": result.get("old_len") or 0,
                "new_len": result.get("new_len") or 0,
                "text": result.get("text") or "",
                "preview": result.get("preview") or "",
                "version": result.get("version") or "",
            }
        )

    # ------------------------------------------------------------------ 指令

    async def _status_text(self) -> str:
        interval = self._cfg_float("interval_days", 1.0)
        lines = ["人格自迭代"]
        scope = await self._scan_persona_scope()
        pm = self.context.persona_manager
        personas_state = self._state.get("personas") or {}
        baseline = float(self._state.get("baseline_ts") or 0)
        if scope:
            lines.append(f"在用的 {len(scope)} 份人格（各自按自己的会话迭代）：")
            for pid, entry in sorted(
                scope.items(), key=lambda kv: float(kv[1].get("latest") or 0), reverse=True
            ):
                persona = pm.get_persona_v3_by_id(pid)
                pstate = personas_state.get(pid) or {}
                last = float(pstate.get("last_run_ts") or 0) or baseline
                next_ts = last + interval * 86400 if last else 0
                lines.append(
                    f"- {pid}：{len(str((persona or {}).get('prompt') or ''))} 字，"
                    f"{len(entry.get('umos') or [])} 个会话，"
                    f"下次 {_fmt_ts(next_ts) if next_ts else '待定'}"
                    f"（{pstate.get('last_result') or '未跑过'}）"
                )
        else:
            target, why = await self._resolve_target()
            if target:
                lines.append(
                    f"目标人格：{target['persona_id']}（{target['source']}，"
                    f"当前 {len(target['prompt'])} 字）"
                )
            else:
                lines.append(f"目标人格：解析失败（{why}）")
        lines.append(
            f"开关：{'开' if self._cfg_bool('enabled', True) else '关'}"
            f"　间隔：{interval:g} 天"
            f"　上限：{self._cfg_int('max_chars', 800)} 字"
            f"　多活跃人格：{'是' if self._cfg_bool('iterate_all_active', True) else '否'}"
        )
        if self._cfg_bool("dry_run", False):
            lines.append("注意：演练模式开启，自动迭代只生成不写入。")
        return "\n".join(lines)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("人格迭代")
    async def persona_evolve_cmd(self, event: AstrMessageEvent):
        """查看状态、立即迭代、预览、历史、回滚。"""
        raw = (event.message_str or "").strip()
        for prefix in ("/", "／", "!"):
            if raw.startswith(prefix):
                raw = raw[len(prefix) :].strip()
        if raw.startswith("人格迭代"):
            raw = raw[len("人格迭代") :].strip()
        parts = raw.split(maxsplit=1)
        action = parts[0].strip() if parts and parts[0].strip() else ""
        rest = parts[1].strip() if len(parts) > 1 else ""

        if action in ("", "状态", "查看"):
            yield event.plain_result(await self._status_text())
            return

        if action in ("立即", "现在", "执行"):
            yield event.plain_result("开始迭代当前人格，等模型返回……")
            result = await self._run_iteration(
                trigger="手动", prefer_umo=event.unified_msg_origin
            )
            if result.get("ok"):
                now = time.time()
                self._state["last_run_ts"] = now
                self._state["last_run_iso"] = _fmt_ts(now)
                self._state["last_result"] = result.get("reason")
                pid = str(result.get("persona_id") or "")
                if pid:
                    pstate = self._state.setdefault("personas", {}).setdefault(pid, {})
                    pstate["last_run_ts"] = now
                    pstate["last_run_iso"] = _fmt_ts(now)
                    pstate["last_result"] = result.get("reason")
                    pstate["fail_count"] = 0
                self._save_state()
            yield event.plain_result(result.get("reason") or "未知结果")
            return

        if action in ("预览", "试跑"):
            result = await self._run_iteration(
                trigger="预览", dry_run=True, prefer_umo=event.unified_msg_origin
            )
            if not result.get("ok"):
                yield event.plain_result(f"预览失败：{result.get('reason')}")
                return
            text = result.get("text") or ""
            body = text if len(text) <= 1500 else text[:1500] + "……"
            yield event.plain_result(
                f"预览（{result.get('persona_id')}，"
                f"{result.get('old_len')} → {result.get('new_len')} 字，未写入）：\n\n{body}"
            )
            return

        if action in ("历史", "版本"):
            target, why = await self._resolve_target()
            if not target:
                yield event.plain_result(f"解析不到当前人格：{why}")
                return
            entries = self._read_index(target["persona_id"])
            if not entries:
                yield event.plain_result("还没有历史版本。")
                return
            lines = [f"{target['persona_id']} 的历史版本（1 为最新）："]
            for idx, entry in enumerate(reversed(entries), start=1):
                lines.append(
                    f"{idx}. {entry.get('ts')}　{entry.get('old_len')}→{entry.get('new_len')} 字"
                    f"　{entry.get('trigger')}"
                )
            yield event.plain_result("\n".join(lines))
            return

        if action in ("回滚", "还原"):
            try:
                index = int(rest) if rest else 1
            except ValueError:
                yield event.plain_result("用法：/人格迭代 回滚 1（1 表示回到上一次迭代之前）")
                return
            target, why = await self._resolve_target()
            if not target:
                yield event.plain_result(f"解析不到当前人格：{why}")
                return
            persona_id = target["persona_id"]
            entries = self._read_index(persona_id)
            if not entries:
                yield event.plain_result("没有可回滚的版本。")
                return
            if index < 1 or index > len(entries):
                yield event.plain_result(f"序号超出范围，共 {len(entries)} 个版本。")
                return
            entry = entries[-index]
            folder = self._version_dir(persona_id)
            try:
                old_text = (folder / str(entry.get("old_file"))).read_text(encoding="utf-8")
            except OSError as exc:
                yield event.plain_result(f"读取版本文件失败：{exc}")
                return
            current = (target["prompt"] or "").strip()
            self._snapshot(persona_id, current, old_text, f"回滚-{index}", "rollback")
            try:
                await self.context.persona_manager.update_persona(
                    persona_id, system_prompt=old_text
                )
            except Exception as exc:
                yield event.plain_result(f"回滚失败：{exc}")
                return
            yield event.plain_result(
                f"已回滚到 {entry.get('ts')} 之前的状态：{len(current)} → {len(old_text)} 字。"
            )
            return

        yield event.plain_result(
            "用法：\n"
            "/人格迭代　查看状态\n"
            "/人格迭代 立即　马上迭代当前人格\n"
            "/人格迭代 预览　只生成不写入\n"
            "/人格迭代 历史　列出历史版本\n"
            "/人格迭代 回滚 1　回到上一次迭代之前"
        )
