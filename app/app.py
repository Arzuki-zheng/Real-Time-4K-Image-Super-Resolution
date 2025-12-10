import streamlit as st
import os
import sys
import subprocess
import tempfile
import shutil
from PIL import Image

# --- 路徑設定 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

st.title("RT4KSR Online Demo")
st.subheader("電腦視覺期末專題 - Real-Time 4K Image Super-Resolution")
st.write("""
組員名單：
    陳鉦元、7114056186
    李品嫻、7114056151
    洪維駿、7114056077
    洪慧珊、7114056078

本示範使用 [Real-Time 4K Image Super-Resolution] 可自行上傳圖片體驗模型效果。
請上傳一張高解析度圖片 (HR)，系統會先將其降階成低解析度圖片 (LR)，再使用 RT4KSR 模型進行超解像 (SR)，最後顯示 SR 結果。
""")
# --- 側邊欄 ---
scale = st.sidebar.selectbox("放大倍率", [2], index=0)
uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 創建一個「系統暫存資料夾」，用完會自動刪除，Streamlit Cloud 允許這裡寫檔
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        # 定義暫存路徑
        hr_path = os.path.join(tmpdirname, "input.png")
        lr_dir = os.path.join(tmpdirname, "LR")
        sr_dir = os.path.join(tmpdirname, "SR")
        
        # 1. 存檔 (把上傳的圖寫入暫存區)
        with open(hr_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(hr_path, caption="原始 HR", width=300)
        
        # 2. 執行 geneLR.py
        # 使用 sys.executable 確保用的是當前環境的 python
        cmd_gene = [
            sys.executable, "geneLR.py",
            "--input", hr_path,      # 直接指定檔案
            "--output", os.path.join(lr_dir, "input.png"), # 指定輸出檔名
            "--scale", str(scale)
        ]
        
        with st.spinner('正在降階 (GeneLR)...'):
            # cwd=PROJECT_ROOT 確保它找得到 geneLR.py
            ret = subprocess.run(cmd_gene, cwd=PROJECT_ROOT, capture_output=True, text=True)
            
            if ret.returncode != 0:
                st.error("geneLR 失敗！")
                st.code(ret.stderr)
            else:
                lr_file = os.path.join(lr_dir, "input.png")
                st.success("LR 產生成功")
                st.image(lr_file, caption="LR 輸入", width=300)

                # 3. 執行 code/test.py
        cmd_test = [
            sys.executable, "code/test.py",
            
            # 根據報錯修正：原本寫 --input 改成 --input_path
            "--input_path", lr_dir,
            
            # 根據報錯修正：原本寫 --output 改成 --output_path
            "--output_path", sr_dir,
            
            "--scale", str(scale),
            
            # 根據報錯修正：拆成 root 和 id
            "--checkpoints-root", os.path.join(PROJECT_ROOT, "code", "checkpoints"),
            "--checkpoint-id", f"rt4ksr_x{scale}",  # 這裡只給檔名 (不含 .pth)
            
            # 其他模型參數 (維持不變)
            "--feature-channels", "24",
            "--num-blocks", "4",
            "--act-type", "gelu",
            "--arch", "rt4ksr_rep"
        ]

        
        with st.spinner('AI 模型推論中...'):
            ret_test = subprocess.run(cmd_test, cwd=PROJECT_ROOT, capture_output=True, text=True)
            
            if ret_test.returncode != 0:
                st.error("Inference 失敗！")
                st.code(ret_test.stderr)
            else:
                # 找出結果
                if os.path.exists(sr_dir) and len(os.listdir(sr_dir)) > 0:
                    sr_filename = os.listdir(sr_dir)[0]
                    sr_path = os.path.join(sr_dir, sr_filename)
                    st.success("SR 推論成功！")
                    st.image(sr_path, caption="SR 結果", use_column_width=True)
                else:
                    st.error("找不到 SR 輸出圖片")

