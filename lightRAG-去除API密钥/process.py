# -*- coding: utf-8 -*-
"""
用 LightRAG 对《加速主义详解.md》做知识图谱索引。

- 输出目录: C:/Users/15076/Desktop/lightrag
- LLM: qwen-plus (DashScope OpenAI 兼容接口)
- Embedding: text-embedding-v3 (1024 维)
- 实体抽取语言: 简体中文

Windows 兼容性处理:
  LightRAG 的多 worker 会并发原子写同一份 JSON(尤其是 kv_store_doc_status.json),
  在 Windows 上 os.replace 遇到目标被瞬时占用(另一 worker / 杀软扫描)会抛
  WinError 5。这里给每个文件的原子写加进程内锁串行化, 并对瞬时锁做带退避的重试。
"""
import asyncio
import os
import time
from functools import partial
from threading import Lock

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

WORKING_DIR = r"C:\Users\15076\Desktop\lightrag"
SOURCE_FILE = r"C:\Users\15076\Desktop\个人创作\加速主义详解.md"

LLM_API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY"))
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", os.environ.get("OPENAI_BASE_URL"))
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-plus")

EMB_API_KEY = os.environ.get("EMB_API_KEY", os.environ.get("OPENAI_API_KEY"))
EMB_BASE_URL = os.environ.get("EMB_BASE_URL", os.environ.get("OPENAI_BASE_URL"))
EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-v3")
EMB_DIM = int(os.environ.get("EMB_DIM", "1024"))

# ---------- Windows 原子写补丁 ----------
_write_locks: dict[str, Lock] = {}
_write_locks_guard = Lock()


def _get_write_lock(path: str) -> Lock:
    norm = os.path.normcase(os.path.abspath(path))
    with _write_locks_guard:
        return _write_locks.setdefault(norm, Lock())


def _make_retrying_atomic_write(original):
    def retrying_atomic_write(file_name, write_fn, workspace="_"):
        with _get_write_lock(file_name):
            last_exc = None
            for attempt in range(10):
                try:
                    original(file_name, write_fn, workspace)
                    return
                except OSError as exc:  # Windows: WinError 5 文件被瞬时占用
                    last_exc = exc
                    time.sleep(0.4 * (attempt + 1))
            raise last_exc

    return retrying_atomic_write


def patch_atomic_write() -> None:
    """atomic_write 被各模块 `from ... import atomic_write` 单独绑定, 需逐一替换。

    某些存储后端(如 faiss)未安装时无法导入, 跳过即可——它们本来也不会被使用。
    """
    mods = [
        "lightrag.utils",
        "lightrag.kg.faiss_impl",
        "lightrag.kg.nano_vector_db_impl",
        "lightrag.kg.networkx_impl",
    ]
    for mod_name in mods:
        try:
            mod = __import__(mod_name, fromlist=["atomic_write"])
        except Exception:
            continue
        if hasattr(mod, "atomic_write"):
            mod.atomic_write = _make_retrying_atomic_write(mod.atomic_write)


def clean_working_dir() -> None:
    """清除上次失败的索引状态; 保留 LLM 响应缓存以复用已支付的实体抽取调用。"""
    keep = {"process.py", "query.py", ".env", "kv_store_llm_response_cache.json"}
    for name in os.listdir(WORKING_DIR):
        path = os.path.join(WORKING_DIR, name)
        if os.path.isfile(path) and name not in keep:
            os.remove(path)


# ---------- LLM / Embedding ----------
async def llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    **kwargs,
) -> str:
    """LightRAG 要求的 llm_model_func 签名是 (prompt, system_prompt, ...)。

    openai_complete_if_cache 的首个位置参数是 model, 这里把 LLM_MODEL 作为
    首个位置参数传入, 避免与框架注入的关键字冲突。
    """
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        **kwargs,
    )


async def main() -> None:
    patch_atomic_write()
    clean_working_dir()

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        llm_model_name=LLM_MODEL,
        addon_params={"language": "Simplified Chinese"},
        embedding_func=EmbeddingFunc(
            embedding_dim=EMB_DIM,
            max_token_size=8192,
            # openai_embed 已被 @wrap_embedding_func_with_attrs 包装, 用 .func 避免双重包装
            func=partial(
                openai_embed.func,
                model=EMB_MODEL,
                base_url=EMB_BASE_URL,
                api_key=EMB_API_KEY,
            ),
        ),
    )

    await rag.initialize_storages()

    with open(SOURCE_FILE, encoding="utf-8") as f:
        content = f.read()

    print(f"[1/2] 读取源文件: {SOURCE_FILE} ({len(content)} 字符)")
    await rag.ainsert(content)
    print("[2/2] 索引完成")

    # 验证: 用混合检索问一个文档里的问题
    res = await rag.aquery(
        "加速主义与技术进步的本质区别是什么？谁才是主体？",
        param=QueryParam(mode="hybrid"),
    )
    print("\n--- 验证查询结果 ---")
    print(res[:500])


if __name__ == "__main__":
    asyncio.run(main())
