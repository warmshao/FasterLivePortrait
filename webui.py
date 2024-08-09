import gradio as gr
import subprocess
import time

# 示例回调函数
def process_image(img_path):
    print("处理中...")  # 在命令行中显示“处理中”

    try:
        # 构造命令行参数
        command = [
            ".\\venv\\python.exe",
            ".\\run.py",
            "--cfg",
            "configs/trt_infer.yaml",
            "--realtime",
            "--dri_video",
            "0",
            "--src_image",
            img_path
        ]
        
        # 执行命令
        result = subprocess.run(command, capture_output=True, text=True)
        
        # 检查命令是否执行成功
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout, stderr=result.stderr)
        
        print("处理完成！")  # 在命令行中显示“处理完成！”
        return img_path, result.stdout
    
    except Exception as e:
        print("处理失败，请检查摄像头并重试！")  # 在命令行中显示错误消息
        print(f"错误详情: {e}")  # 打印错误详情以帮助调试
        return None, "处理失败，请检查摄像头并重试！"

examples_path = "assets/examples/source"

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("""
    <div style="text-align: center; font-size: 24px; font-weight: bold;">
    实时面部控制交互  
    </div>
    
    1. 请确保摄像头启用，并使摄像头画面中有人脸。  
    2. 选择一张图像，点击'开始处理'按钮，后台会显示“处理中”。  
    3. 若后台显示处理失败，请检查摄像头是否正常工作，人脸角度位置，并重试。  
    4. 若想要结束处理，在render窗口中按下Q键退出窗口，等待后台显示“处理完成”后再进行下一步操作。
    """)
    
    # 限制上传图片窗口的尺寸
    img_input = gr.Image(type="filepath", label="选择图像", height=512, width=512)  # 限制窗口为512x512像素
    
    # 自定义按钮
    with gr.Row():
        submit_btn = gr.Button("开始处理")
    
    # 将自定义按钮与处理函数连接
    submit_btn.click(process_image, inputs=img_input)
    
    # 示例图片
    gr.Examples(examples=[f"{examples_path}/s5.jpg",
                          f"{examples_path}/s7.jpg",
                          f"{examples_path}/s39.jpg"],
                inputs=img_input)

demo.launch(inbrowser=True,
     server_port=9871,
     share=False,
     server_name="127.0.0.1"
)

if __name__ == "__main__":
    gradio_interface()
