from langchain_core.callbacks.base import AsyncCallbackHandler
import asyncio


class SSECallbackHandler(AsyncCallbackHandler):
    """自定义异步回调处理器"""

    def __init__(self, queue: asyncio.Queue):
        super().__init__()
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """每个 token 生成时触发 - 最重要！"""
        if token:
            await self.queue.put({
                'type': 'token',
                'content': token
            })

    async def on_tool_start(
            self,
            serialized: dict,
            input_str: str,
            **kwargs
    ) -> None:
        """工具调用开始"""
        #tool_name = serialized.get('name', 'unknown')
        # await self.queue.put({
        #     'type': 'tool_start',
        #     'tool': tool_name,
        #     'input': input_str
        # })

    async def on_tool_end(self, output: str, **kwargs) -> None:
        """工具调用结束"""
        # await self.queue.put({
        #     'type': 'tool_end',
        #     'output': output
        # })

    async def on_tool_error(self, error: Exception, **kwargs) -> None:
        """工具错误处理"""
        print(f"[QUEUE PUT] tool error error={error}")
        # await self.queue.put({
        #     'type': 'error',
        #     'message': f"Tool error: {str(error)}"
        # })

    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        """LLM 错误处理"""
        print(f"[QUEUE PUT] llm error error={error}")
        # await self.queue.put({
        #     'type': 'error',
        #     'message': str(error)
        # })

    async def on_chain_end(self, output, **kwargs) -> None:
        """链结束时触发"""
        #print(f"[QUEUE PUT] chain end type={type(output)}, outputs={output}")
        # await self.queue.put({
        #     'type': 'chain_end',
        #     'outputs': output
        # })

    async def on_chain_error(self, error, **kwargs) -> None:
        """链错误时触发"""
        print(f"[QUEUE PUT] chain error error={repr(error)}", flush=True)
        traceback.print_exc(file=sys.stdout)
        # await self.queue.put({
        #     'type': 'error',
        #     'message': str(error),
        #     'details': kwargs  # 这里可能包含复杂对象
        # })

    async def on_agent_finish(self, finish, **kwargs):
        await self.queue.put({
            'type': 'end',
            'content': ''
        })