#!/usr/bin/env python3
"""
QwenTTS 零样本语音克隆 API 测试套件。

前提条件：
  1. TTS 服务器已启动（默认 127.0.0.1:8080）
  2. 使用 base 模型（如 qwen-talker-1.7b-base）
     - base 模型：支持零样本克隆（ref_wav_b64 / ref_wav_path）
     - custom_voice 模型：支持固定音色（voice 参数），不支持零样本
     - voice_design 模型：支持风格指令（instructions），不支持零样本
  3. 测试资源文件存在于 asset/ 和 examples/ 目录

运行方式：
  python test.py
"""

import requests
import json
import base64
import sys
import os
from pathlib import Path

# ── 服务器配置 ──
API_BASE_URL = "http://127.0.0.1:8080"
ASSET_DIR = Path(__file__).parent / "asset"
EXAMPLES_DIR = Path(__file__).parent / "examples"
OUTPUT_DIR = Path(__file__).parent / "test_output"

# 创建测试输出目录
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 测试资源路径 ──
REF_WAV = ASSET_DIR / "ref.wav"
REF_TXT = ASSET_DIR / "ref.txt"
FREEMAN_WAV = EXAMPLES_DIR / "freeman.wav"
FREEMAN_TXT = EXAMPLES_DIR / "freeman.txt"


def load_ref_text(txt_path):
    """加载参考音频对应的文本转录。"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_wav_b64(wav_path):
    """读取 WAV 文件并编码为 Base64 字符串。"""
    with open(wav_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_zero_shot_b64():
    """零样本合成 — Base64 参考音频模式（需要 base 模型）。

    通过 ref_wav_b64 字段传入 Base64 编码的参考音频，
    配合 ref_text 提供参考音频的文本转录。
    服务器会将参考音频编码为说话人嵌入和 RVQ 码，用于零样本克隆。
    """
    print("\n" + "="*60)
    print("Test 1: Zero-shot synthesis with Base64 reference audio")
    print("="*60)
    
    if not REF_WAV.exists() or not REF_TXT.exists():
        print(f"❌ Missing assets: {REF_WAV} or {REF_TXT}")
        return False
    
    try:
        ref_text = load_ref_text(REF_TXT)
        ref_wav_b64 = load_wav_b64(REF_WAV)
        
        print(f"✓ Loaded reference audio: {REF_WAV}")
        print(f"✓ Loaded reference text: {ref_text[:50]}...")
        print(f"✓ Base64 encoded (length: {len(ref_wav_b64)})")
        
        # 客户端侧验证：打印 Base64 前后各 50 字符，用于调试
        print(f"  Base64 前 50 字符: {ref_wav_b64[:50]}")
        print(f"  Base64 后 50 字符: {ref_wav_b64[-50:]}")
        
        # 验证 Python 侧 Base64 解码是否正常（与服务端解码独立）
        import base64
        try:
            decoded = base64.b64decode(ref_wav_b64)
            print(f"  ✓ Base64 解码成功，大小: {len(decoded)} 字节")
            # 检查 WAV 文件头（RIFF + WAVE）
            if decoded[:4] == b'RIFF' and decoded[8:12] == b'WAVE':
                print(f"  ✓ 解码后为有效的 WAV 文件")
            else:
                print(f"  ❌ 解码后不是有效的 WAV 文件")
                print(f"     头部: {decoded[:12]}")
        except Exception as e:
            print(f"  ❌ Base64 解码失败: {e}")
        
        payload = {
            "input": "我很高兴见到你。这是一个测试。",
            "ref_wav_b64": ref_wav_b64,
            "ref_text": ref_text,
            "response_format": "wav"
        }
        
        print(f"\n📤 Sending request to {API_BASE_URL}/v1/audio/speech...")
        print(f"   Input: {payload['input']}")
        print(f"   Response format: wav")
        
        response = requests.post(
            f"{API_BASE_URL}/v1/audio/speech",
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        output_file = OUTPUT_DIR / "test_zero_shot_b64.wav"
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Success! Generated audio saved to: {output_file}")
        print(f"   File size: {output_file.stat().st_size} bytes")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_zero_shot_path():
    """零样本合成 — 文件路径模式，使用 examples/freeman.wav（需要 base 模型）。

    通过 ref_wav_path 字段传入服务器可读的绝对路径，
    避免 Base64 编解码开销，适合大文件参考音频。
    """
    print("\n" + "="*60)
    print("Test 2: Zero-shot synthesis with file path (freeman.wav)")
    print("="*60)
    
    if not FREEMAN_WAV.exists() or not FREEMAN_TXT.exists():
        print(f"❌ Missing assets: {FREEMAN_WAV} or {FREEMAN_TXT}")
        return False
    
    try:
        ref_text = load_ref_text(FREEMAN_TXT)
        abs_path = str(FREEMAN_WAV.absolute())
        
        print(f"✓ Reference audio path: {abs_path}")
        print(f"✓ Loaded reference text: {ref_text[:50]}...")
        
        payload = {
            "input": "This is a test of zero-shot voice cloning.",
            "ref_wav_path": abs_path,
            "ref_text": ref_text,
            "response_format": "wav"
        }
        
        print(f"\n📤 Sending request to {API_BASE_URL}/v1/audio/speech...")
        print(f"   Input: {payload['input']}")
        print(f"   Response format: wav")
        
        response = requests.post(
            f"{API_BASE_URL}/v1/audio/speech",
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        output_file = OUTPUT_DIR / "test_zero_shot_path_freeman.wav"
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Success! Generated audio saved to: {output_file}")
        print(f"   File size: {output_file.stat().st_size} bytes")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_zero_shot_path_ref():
    """零样本合成 — 文件路径模式，使用 asset/ref.wav（需要 base 模型）。

    与 test_zero_shot_path 相同，但使用不同的参考音频，
    验证不同音频文件的克隆效果和缓存独立性。
    """
    print("\n" + "="*60)
    print("Test 2b: Zero-shot synthesis with file path (asset/ref.wav)")
    print("="*60)
    
    if not REF_WAV.exists() or not REF_TXT.exists():
        print(f"❌ Missing assets: {REF_WAV} or {REF_TXT}")
        return False
    
    try:
        ref_text = load_ref_text(REF_TXT)
        abs_path = str(REF_WAV.absolute())
        
        print(f"✓ Reference audio path: {abs_path}")
        print(f"✓ Loaded reference text: {ref_text[:50]}...")
        
        payload = {
            "input": "我很高兴见到你。这是一个测试。",
            "ref_wav_path": abs_path,
            "ref_text": ref_text,
            "response_format": "wav"
        }
        
        print(f"\n📤 Sending request to {API_BASE_URL}/v1/audio/speech...")
        print(f"   Input: {payload['input']}")
        print(f"   Response format: wav")
        
        response = requests.post(
            f"{API_BASE_URL}/v1/audio/speech",
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        output_file = OUTPUT_DIR / "test_zero_shot_path_ref.wav"
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Success! Generated audio saved to: {output_file}")
        print(f"   File size: {output_file.stat().st_size} bytes")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_encode_endpoint():
    """预编码端点 /v1/audio/encode（需要 base 模型）。

    将参考音频预编码为说话人嵌入（spk）和 RVQ 码本（rvq），
    编码结果按内容哈希缓存到 qwentts_server_cache/ 目录。
    相同音频重复调用会命中缓存，返回 cached: true。
    ref_text 为必填字段，与编码结果一同保存为 {hash}.txt。
    """
    print("\n" + "="*60)
    print("Test 3: /v1/audio/encode pre-encoding endpoint")
    print("="*60)
    
    if not REF_WAV.exists() or not REF_TXT.exists():
        print(f"❌ Missing assets: {REF_WAV} or {REF_TXT}")
        return False
    
    try:
        ref_text = load_ref_text(REF_TXT)
        ref_wav_b64 = load_wav_b64(REF_WAV)
        
        print(f"✓ Loaded reference audio: {REF_WAV}")
        print(f"✓ Loaded reference text: {ref_text[:50]}...")
        print(f"✓ Base64 encoded (length: {len(ref_wav_b64)})")
        
        payload = {
            "ref_wav_b64": ref_wav_b64,
            "ref_text": ref_text
        }
        
        print(f"\n📤 Sending request to {API_BASE_URL}/v1/audio/encode...")
        
        response = requests.post(
            f"{API_BASE_URL}/v1/audio/encode",
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        result = response.json()
        print(f"✓ Success! Pre-encoding result:")
        print(f"   - Hash: {result.get('hash', 'N/A')}")
        print(f"   - Cached: {result.get('cached', False)}")
        print(f"   - Speaker embedding (.spk): {len(result.get('spk', ''))} chars (base64)")
        print(f"   - RVQ codebook (.rvq): {len(result.get('rvq', ''))} chars (base64)")
        
        # 保存编码结果到 JSON（含 hash、spk、rvq、ref_text、cached 字段）
        output_file = OUTPUT_DIR / "test_encode_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✓ Result saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_hit():
    """缓存命中测试（需要 base 模型）。

    发送两次相同的零样本合成请求：
    第一次：缓存未命中 → 编码参考音频 → 保存到缓存 → 合成
    第二次：缓存命中 → 加载预编码 → 直接合成
    预期第二次请求更快（跳过编码步骤）。
    """
    print("\n" + "="*60)
    print("Test 4: Cache hit (repeated request should be faster)")
    print("="*60)
    
    if not REF_WAV.exists() or not REF_TXT.exists():
        print(f"❌ Missing assets: {REF_WAV} or {REF_TXT}")
        return False
    
    try:
        ref_text = load_ref_text(REF_TXT)
        ref_wav_b64 = load_wav_b64(REF_WAV)
        
        payload = {
            "input": "缓存测试",
            "ref_wav_b64": ref_wav_b64,
            "ref_text": ref_text,
            "response_format": "wav"
        }
        
        import time
        
        # 第一次请求：缓存未命中，需完整编码
        print("\n📤 First request (cache miss)...")
        start_time = time.time()
        response1 = requests.post(
            f"{API_BASE_URL}/v1/audio/speech",
            json=payload,
            timeout=60
        )
        time1 = time.time() - start_time
        
        if response1.status_code != 200:
            print(f"❌ First request failed: HTTP {response1.status_code}")
            return False
        
        print(f"✓ First request completed in {time1:.2f}s")
        
        # 第二次请求：缓存命中，预期更快
        print("\n📤 Second request (cache hit)...")
        start_time = time.time()
        response2 = requests.post(
            f"{API_BASE_URL}/v1/audio/speech",
            json=payload,
            timeout=60
        )
        time2 = time.time() - start_time
        
        if response2.status_code != 200:
            print(f"❌ Second request failed: HTTP {response2.status_code}")
            return False
        
        print(f"✓ Second request completed in {time2:.2f}s")
        
        if time2 < time1:
            print(f"✅ Cache working! Second request was {time1/time2:.1f}x faster")
        else:
            print(f"⚠️  Second request was not faster (cache may not be enabled)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_regular_synthesis():
    """常规 TTS 合成测试（需要 custom_voice 模型）。

    不使用零样本克隆，仅通过 voice 参数选择固定音色。
    注意：此测试在 base 模型下预期失败，因为 base 模型无预设音色。
    """
    print("\n" + "="*60)
    print("Test 5: Regular TTS synthesis (non-zero-shot)")
    print("="*60)
    
    try:
        # voice 参数仅在 custom_voice 模型下生效，需使用 GET /v1/voices 返回的有效音色名
        # base 模型无预设音色，voice 会被静默忽略，此时输出音色由随机采样决定
        payload = {
            "input": "这是一个常规合成测试。",
            "voice": "default",  # 需替换为 custom_voice 模型的实际音色名
            "response_format": "wav"
        }
        
        print(f"📤 发送常规 TTS 请求...")
        print(f"   输入文本: {payload['input']}")
        print(f"   音色: {payload['voice']}（仅 custom_voice 模型生效）")
        
        response = requests.post(
            f"{API_BASE_URL}/v1/audio/speech",
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"⚠️  常规合成不可用: HTTP {response.status_code}（base 模型下预期失败）")
            return False
        
        output_file = OUTPUT_DIR / "test_regular_synthesis.wav"
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Success! Generated audio saved to: {output_file}")
        print(f"   File size: {output_file.stat().st_size} bytes")
        return True
        
    except Exception as e:
        print(f"⚠️  Regular synthesis test failed: {e}")
        return False


def test_health():
    """服务器健康检查。确认 TTS 服务器已启动并可访问。"""
    print("\n" + "="*60)
    print("Test 0: Server health check")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ Server is healthy")
            return True
        else:
            print(f"❌ Server returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print(f"\n请确保 TTS 服务器已启动：")
        print(f"  ./build/tts-server --model models/qwen-talker-1.7b-base-Q8_0.gguf \\")
        print(f"                     --codec models/qwen-tokenizer-12hz-Q8_0.gguf")
        return False


def main():
    """运行所有测试用例。"""
    print("\n" + "="*60)
    print("QwenTTS 零样本语音克隆 API 测试套件")
    print("="*60)
    print("  前提：服务器已启动，使用 base 模型")
    print("  输出目录：", OUTPUT_DIR.absolute())
    
    # 先检查服务器是否在线
    if not test_health():
        sys.exit(1)
    
    # 按顺序执行测试
    results = {}
    results["health"] = test_health()
    results["zero_shot_b64"] = test_zero_shot_b64()
    results["zero_shot_path_freeman"] = test_zero_shot_path()
    results["zero_shot_path_ref"] = test_zero_shot_path_ref()
    results["encode"] = test_encode_endpoint()
    results["cache_hit"] = test_cache_hit()
    results["regular_synthesis"] = test_regular_synthesis()
    
    # 输出测试结果汇总
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n通过: {passed}/{total} 个测试")
    print(f"输出目录: {OUTPUT_DIR.absolute()}")
    print(f"\n注意: regular_synthesis 测试在 base 模型下预期失败，属于正常现象")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
