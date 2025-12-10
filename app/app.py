import streamlit as st
import os
import sys
import subprocess
import tempfile
import shutil
from PIL import Image

# --- 1. 設定頁面為寬版模式 (這行必須是第一行 Streamlit 指令) ---
st.set_page_config(page_title="RT4KSR Demo", layout="wide")

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
    # 創建一個「系統暫存資料夾」
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        # 定義暫存路徑
        hr_path = os.path.join(tmpdirname, "input.png")
        lr_dir = os.path.join(tmpdirname, "LR")
        sr_dir = os.path.join(tmpdirname, "SR")
        
        # 1. 存檔
        with open(hr_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 2. 執行 geneLR.py
        cmd_gene = [
            sys.executable, "geneLR.py",
            "--input", hr_path,
            "--output", os.path.join(lr_dir, "input.png"),
            "--scale", str(scale)
        ]
        
        with st.spinner('正在降階 (GeneLR)...'):
            ret = subprocess.run(cmd_gene, cwd=PROJECT_ROOT, capture_output=True, text=True)
            if ret.returncode != 0:
                st.error("geneLR 失敗！")
                st.code(ret.stderr)
            else:
                lr_file = os.path.join(lr_dir, "input.png")
                # 這裡先把 LR 路徑存起來，等一下一起顯示

        # 3. 執行 code/test.py
        cmd_test = [
            sys.executable, "code/test.py",
            "--input_path", lr_dir,
            "--output_path", sr_dir,
            "--scale", str(scale),
            "--checkpoints-root", os.path.join(PROJECT_ROOT, "code", "checkpoints"),
            "--checkpoint-id", f"rt4ksr_x{scale}",
            "--feature-channels", "24",
            "--num-blocks", "4",
            "--act-type", "gelu",
            "--arch", "rt4ksr_rep",
            "--is-train"  # 關鍵修正
        ]
        
        with st.spinner('AI 模型推論中 (RT4KSR)...'):
            ret_test = subprocess.run(cmd_test, cwd=PROJECT_ROOT, capture_output=True, text=True)
            
            if ret_test.returncode != 0:
                st.error("Inference 失敗！")
                st.code(ret_test.stderr)
            else:
                if os.path.exists(sr_dir) and len(os.listdir(sr_dir)) > 0:
                    sr_filename = os.listdir(sr_dir)[0]
                    sr_path = os.path.join(sr_dir, sr_filename)
                    st.success("推論完成！")
                    
                    st.divider() # 分隔線
                    
                    # --- Part A: 三張圖並排顯示 (Overview) ---
                    st.subheader("1. 流程總覽 (Overview)")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("##### 原始 HR (Ground Truth)")
                        st.image(hr_path, use_column_width=True)
                        
                    with col2:
                        st.write(f"##### 降階 LR (Input x{scale})")
                        # 這裡放 LR (通常比較糊)
                        st.image(lr_file, use_column_width=True)
                        
                    with col3:
                        st.write(f"##### 模型 SR (Result x{scale})")
                        st.image(sr_path, use_column_width=True)
                    
                    st.divider() # 分隔線

                    # --- Part B: 大圖對比 (Compare) ---
                    st.subheader("2. 細節對比 (Compare: HR vs SR)")
                    st.info("比較細節差異")
                    
                    comp_col1, comp_col2 = st.columns(2)
                    with comp_col1:
                        st.write("### 原始 HR")
                        st.image(hr_path, use_column_width=True)
                    
                    with comp_col2:
                        st.write("### 模型 SR")
                        st.image(sr_path, use_column_width=True)

                else:
                    st.error("找不到 SR 輸出圖片")
