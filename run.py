import asyncio
import logging
import os
import json
import sqlite3
import time
import base64
import io
import aiohttp
import tempfile

from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.filters.command import CommandStart
from aiogram.utils.keyboard import InlineKeyboardBuilder
from faster_whisper import WhisperModel

# ====== 環境變數 ======
TOKEN     = os.getenv("TOKEN")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
INITMODEL   = os.getenv("INITMODEL", "qwen2.5:3b")
ADMIN_IDS   = (
    list(map(int, os.getenv("ADMIN_IDS", "").split(",")))
    if os.getenv("ADMIN_IDS") else []
)

# ====== 初始化 Bot ======
bot = Bot(token=TOKEN)
dp  = Dispatcher()
logging.basicConfig(level=logging.INFO)

modelname        = INITMODEL
MODEL_TEMP_CACHE = {}

# ====== Whisper 初始化 ======
logging.info("⏳ 載入 Whisper small 模型中...")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
logging.info("✅ Whisper 載入完成")

# ====== DB ======
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)")
    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER,
            role      TEXT,
            content   TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_chat(user_id: int, role: str, content: str):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO chats (user_id, role, content) VALUES (?, ?, ?)",
        (user_id, role, content)
    )
    conn.commit()
    conn.close()

# ====== 權限 ======
def is_allowed(user_id: int) -> bool:
    return (not ADMIN_IDS) or (user_id in ADMIN_IDS)

# ====== Ollama ======
async def generate(messages: list, model: str):
    url = f"http://127.0.0.1:{OLLAMA_PORT}/api/chat"
    # 增加連接超時設定，避免無限等待
    timeout = aiohttp.ClientTimeout(total=600, connect=10, sock_read=600)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url, json={
                "model":    model,
                "messages": messages,
                "stream":   True
            }) as resp:
                buffer = b""
                async for chunk in resp.content.iter_any():
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if line.strip():
                            yield json.loads(line)
        except asyncio.TimeoutError:
            logging.error("[generate] Ollama 請求超時")
            raise
        except Exception as e:
            logging.error(f"[generate] 請求錯誤：{e}")
            raise

async def model_list() -> list:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags") as r:
            data = await r.json()
            return data.get("models", [])

# ====== 輸入處理 ======
async def process_image(message: types.Message) -> str:
    photo = None
    if message.photo:
        photo = message.photo[-1]
    elif message.document and message.document.mime_type and \
            message.document.mime_type.startswith("image/"):
        photo = message.document

    if photo:
        buffer = io.BytesIO()
        await message.bot.download(photo, destination=buffer)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    return ""

def _whisper_transcribe(path: str) -> str:
    segments, _ = whisper_model.transcribe(path, language="zh", beam_size=5)
    return "".join(s.text for s in segments).strip()

async def process_voice(message: types.Message) -> str:
    try:
        buffer = io.BytesIO()
        await message.bot.download(message.voice, destination=buffer)
        buffer.seek(0)
        tmp_path = os.path.join(tempfile.gettempdir(), "tg_voice.ogg")
        with open(tmp_path, "wb") as f:
            f.write(buffer.read())
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, _whisper_transcribe, tmp_path)
        logging.info(f"[Whisper] 辨識結果：{text}")
        return text
    except Exception as e:
        logging.error(f"[process_voice] 錯誤：{e}")
        return ""

# ====== 回覆工具 ======
async def safe_send(message: types.Message, text: str):
    try:
        await message.answer(text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        try:
            await message.answer(text)
        except Exception:
            await message.answer(f"內容格式錯誤：\n{text[:1000]}...")

async def safe_edit(message: types.Message, text: str, msg_id: int):
    try:
        await bot.edit_message_text(text=text, chat_id=message.chat.id, message_id=msg_id, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        if "message is not modified" not in str(e).lower():
            # 如果是速率限制，記錄但不崩潰
            if "Flood control" in str(e):
                logging.warning(f"[safe_edit] 觸發速率限制，跳過本次更新")
            else:
                logging.warning(f"[safe_edit] 編輯失敗：{e}")

# ====== 指令：/start ======
@dp.message(CommandStart())
async def cmd_start(msg: types.Message):
    text = f"""🤖 *AI Bot 已啟動（繁體中文模式）*

歡迎使用本機 Ollama AI 助手 🚀

📌 *支援輸入方式*
- 💬 純文字
- 🖼 圖片（含說明文字）
- 🎤 語音訊息（自動辨識轉文字）
- 🖼🎤 圖片 ＋ 語音（同時理解圖與語音）

🧠 *模型管理*
✔ `/models` → 切換模型

⚙️ *目前模型*：`{modelname}`

開始輸入問題吧 👇"""
    await msg.answer(text, parse_mode=ParseMode.MARKDOWN)

# ====== 指令：/models ======
@dp.message(lambda m: m.text and m.text.strip() == "/models")
async def cmd_models(msg: types.Message):
    models = await model_list()
    if not models:
        return await msg.answer("❌ 找不到任何模型，請確認 Ollama 是否正在執行")
    kb = InlineKeyboardBuilder()
    MODEL_TEMP_CACHE.clear()
    for i, m in enumerate(models):
        name = m["name"]
        MODEL_TEMP_CACHE[str(i)] = name
        kb.row(types.InlineKeyboardButton(text=name, callback_data=f"model_{i}"))
    await msg.answer("📦 選擇模型：", reply_markup=kb.as_markup())

@dp.callback_query(lambda c: c.data and c.data.startswith("model_"))
async def cb_switch_model(cb: types.CallbackQuery):
    global modelname
    idx  = cb.data.split("_")[1]
    name = MODEL_TEMP_CACHE.get(idx)
    if name:
        modelname = name
        await cb.message.edit_text(f"✅ 已切換模型：`{name}`", parse_mode=ParseMode.MARKDOWN)
    else:
        await cb.answer("選項已失效，請重新執行 /models", show_alert=True)

# ====== 主訊息處理 ======
@dp.message()
async def handle(msg: types.Message):
    if not is_allowed(msg.from_user.id):
        return await msg.answer("❌ 無使用權限")
    await ollama_request(msg)

async def _waiting_ticker(message: types.Message, start_time: float, ack_msg_id: int, stop_event: asyncio.Event):
    """每 3 秒更新一次等待狀態，並監聽停止信號"""
    count = 0
    model_display = modelname.split('/')[-1] if '/' in modelname else modelname
    logging.info(f"[ticker] 計時器啟動，模型：{model_display}")
    
    try:
        while not stop_event.is_set():
            try:
                # 使用 wait_for 確保能即時響應停止信號
                await asyncio.wait_for(stop_event.wait(), timeout=3.0)
                break # 如果 wait_for 完成，表示 stop_event 被設置了
            except asyncio.TimeoutError:
                pass # 正常超時，繼續執行
            
            if stop_event.is_set():
                break
                
            elapsed = int(time.time() - start_time)
            text = f"⏳ 💬 文字 處理中，模型：{model_display}\n已等待 {elapsed} 秒…"
            await safe_edit(message, text, ack_msg_id)
            count += 1
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logging.error(f"[ticker] 錯誤：{e}")
    finally:
        logging.info(f"[ticker] 計時器結束，總共更新 {count} 次")

async def ollama_request(message: types.Message):
    ticker_task = None
    stop_event = None
    
    try:
        # 1. 取得文字 prompt
        prompt = message.text or message.caption or ""

        # 2. 語音處理
        voice_text = ""
        if message.voice:
            await message.answer("🎤 語音辨識中，請稍候...")
            voice_text = await process_voice(message)
            if not voice_text:
                return await message.answer("❌ 語音辨識失敗，請重試或改用文字輸入")
            await message.answer(f"📝 *語音辨識結果：*\n{voice_text}", parse_mode=ParseMode.MARKDOWN)
            prompt = (prompt + "\n" + voice_text).strip() if prompt else voice_text

        # 3. 圖片處理
        img_b64 = await process_image(message)

        # 4. 判斷輸入組合
        has_text  = bool(prompt)
        has_image = bool(img_b64)
        if not has_text and not has_image:
            return await message.answer("❓ 請傳送文字、語音或圖片（可以組合使用）")
        if has_image and not has_text:
            prompt = "請詳細描述這張圖片的內容。"

        input_types = []
        if has_image: input_types.append("🖼 圖片")
        if message.voice: input_types.append("🎤 語音")
        elif message.text or message.caption: input_types.append("💬 文字")
        input_label = " ＋ ".join(input_types)

        # 5. ACK (初始訊息)
        model_display = modelname.split('/')[-1] if '/' in modelname else modelname
        ack_text = f"⏳ 💬 {input_label} 處理中，模型：{model_display}\n已等待 0 秒…"
        ack_msg = await message.answer(ack_text, parse_mode=ParseMode.MARKDOWN)
        ack_msg_id = ack_msg.message_id

        # 6. 組裝訊息
        start_time = time.time()
        system_prompt = "請一律使用繁體中文回答，並且回答清楚。"
        user_content = {"role": "user", "content": prompt}
        if img_b64: user_content["images"] = [img_b64]
        msgs = [{"role": "system", "content": system_prompt}, user_content]

        # 7. 啟動計時器
        stop_event = asyncio.Event()
        ticker_task = asyncio.create_task(_waiting_ticker(message, start_time, ack_msg_id, stop_event))

        # 8. 串流生成 (加入整體超時保護)
        full = ""
        try:
            # 設定最長等待時間為 10 分鐘，避免無限卡死
            async with asyncio.timeout(600): 
                async for resp in generate(msgs, modelname):
                    chunk = resp.get("message", {}).get("content", "")
                    full += chunk
                    if resp.get("done"):
                        break
        except asyncio.TimeoutError:
            logging.error("[ollama_request] 整體請求超時 (10分鐘)")
            raise Exception("Ollama 回應超時，模型處理時間過長")

        # 9. 成功完成：停止計時器並發送結果
        if stop_event: stop_event.set()
        if ticker_task:
            try: await ticker_task
            except Exception: pass
        
        elapsed = time.time() - start_time
        reply = f"{full}\n\n⚙️ 模型：{modelname}\n⏱️ 時間：{elapsed:.2f} 秒"
        await safe_send(message, reply)
        save_chat(message.from_user.id, "assistant", full)

    except Exception as e:
        logging.error(f"[ollama_request] 錯誤：{e}")
        # 發生錯誤時強制停止計時器
        if stop_event: stop_event.set()
        if ticker_task:
            try: 
                # 給予短暫時間讓計時器任務清理
                await asyncio.wait_for(ticker_task, timeout=2.0)
            except asyncio.TimeoutError:
                ticker_task.cancel()
            except Exception: pass
        
        await message.answer("❌ 發生錯誤，請稍後再試")

# ====== 啟動 ======
async def main():
    init_db()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
