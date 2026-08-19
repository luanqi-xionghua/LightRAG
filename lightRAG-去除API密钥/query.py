# -*- coding: utf-8 -*-
"""
查询已建好的 LightRAG 知识图谱。

用法:
  1) 单次查询:  python query.py "加速主义有哪些派别？"
  2) 交互模式:  python query.py            (输入问题, 输入 q 退出)
  3) 指定模式:  python query.py "问题" --mode global

mode 可选: hybrid(默认, 图谱+向量) / global(全局, 跨文档) / local(局部) / naive(纯向量)
"""
import asyncio
import os
import sys
from functools import partial

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

WORKING_DIR = r"C:\Users\15076\Desktop\lightrag"
LLM_API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY"))
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", os.environ.get("OPENAI_BASE_URL"))
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-plus")

EMB_API_KEY = os.environ.get("EMB_API_KEY", os.environ.get("OPENAI_API_KEY"))
EMB_BASE_URL = os.environ.get("EMB_BASE_URL", os.environ.get("OPENAI_BASE_URL"))
EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-v3")
EMB_DIM = int(os.environ.get("EMB_DIM", "1024"))


async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    return await openai_complete_if_cache(
        LLM_MODEL, prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        **kwargs,
    )


async def ask(rag, question: str, mode: str) -> None:
    print(f"\n【问题】{question}")
    print(f"【模式】{mode}")
    print("=" * 60)
    res = await rag.aquery(question, param=QueryParam(mode=mode))
    print(res)
    print("=" * 60)


async def main() -> None:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        llm_model_name=LLM_MODEL,
        addon_params={"language": "Simplified Chinese"},
        embedding_func=EmbeddingFunc(
            embedding_dim=EMB_DIM,
            max_token_size=8192,
            func=partial(
                openai_embed.func,
                model=EMB_MODEL,
                base_url=EMB_BASE_URL,
                api_key=EMB_API_KEY,
            ),
        ),
    )
    await rag.initialize_storages()

    args = sys.argv[1:]
    mode = "hybrid"
    if "--mode" in args:
        i = args.index("--mode")
        mode = args[i + 1]
        args = args[:i] + args[i + 2:]

    if args:
        await ask(rag, args[0], mode)
    else:
        print("交互模式：直接输入问题，输入 q 退出。")
        while True:
            try:
                q = input("\n问题 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() == "q":
                break
            await ask(rag, q, mode)


if __name__ == "__main__":
    asyncio.run(main())
