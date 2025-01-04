# coding: utf-8

"""
The entrance of the gradio
"""
import pdb

import gradio as gr
import os.path as osp
from omegaconf import OmegaConf

from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline


def load_description(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


import argparse

parser = argparse.ArgumentParser(description='Faster Live Portrait Pipeline')
parser.add_argument('--mode', required=False, type=str, default="onnx")
parser.add_argument('--use_mp', action='store_true', help='use mediapipe or not')
parser.add_argument(
    "--host_ip", type=str, default="127.0.0.1", help="host ip"
)
parser.add_argument("--port", type=int, default=9870, help="server port")
args, unknown = parser.parse_known_args()

if args.mode == "onnx":
    cfg_path = "configs/onnx_mp_infer.yaml" if args.use_mp else "configs/onnx_infer.yaml"
else:
    cfg_path = "configs/trt_mp_infer.yaml" if args.use_mp else "configs/trt_infer.yaml"
infer_cfg = OmegaConf.load(cfg_path)
gradio_pipeline = GradioLivePortraitPipeline(infer_cfg)


def gpu_wrapped_execute_video(*args, **kwargs):
    return gradio_pipeline.execute_video(*args, **kwargs)


def gpu_wrapped_execute_image(*args, **kwargs):
    return gradio_pipeline.execute_image(*args, **kwargs)


def change_animal_model(is_animal):
    global gradio_pipeline
    gradio_pipeline.clean_models()
    gradio_pipeline.init_models(is_animal=is_animal)


# assets
title_md = "assets/gradio/gradio_title.md"
example_portrait_dir = "assets/examples/source"
example_video_dir = "assets/examples/driving"
#################### interface logic ####################

# Define components first
eye_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target eyes-open ratio")
lip_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target lip-open ratio")
retargeting_input_image = gr.Image(type="filepath")
output_image = gr.Image(format="png", type="numpy")
output_image_paste_back = gr.Image(format="png", type="numpy")

js_func = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """

with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")]), js=js_func) as demo:
    gr.HTML(load_description(title_md))

    gr.Markdown(load_description("assets/gradio/gradio_description_upload.md"))
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("🖼️ Source Image") as tab_image:
                    with gr.Accordion(open=True, label="Source Image"):
                        source_image_input = gr.Image(type="filepath")
                        gr.Examples(
                            examples=[
                                [osp.join(example_portrait_dir, "s9.jpg")],
                                [osp.join(example_portrait_dir, "s6.jpg")],
                                [osp.join(example_portrait_dir, "s10.jpg")],
                                [osp.join(example_portrait_dir, "s5.jpg")],
                                [osp.join(example_portrait_dir, "s7.jpg")],
                                [osp.join(example_portrait_dir, "s12.jpg")],
                            ],
                            inputs=[source_image_input],
                            cache_examples=False,
                        )

                with gr.TabItem("🎞️ Source Video") as tab_video:
                    with gr.Accordion(open=True, label="Source Video"):
                        source_video_input = gr.Video()
                        gr.Examples(
                            examples=[
                                [osp.join(example_video_dir, "d9.mp4")],
                                [osp.join(example_video_dir, "d10.mp4")],
                                [osp.join(example_video_dir, "d11.mp4")],
                                [osp.join(example_video_dir, "d12.mp4")],
                                [osp.join(example_video_dir, "d13.mp4")],
                                [osp.join(example_video_dir, "d14.mp4")],
                            ],
                            inputs=[source_video_input],
                            cache_examples=False,
                        )

                tab_selection = gr.Textbox(visible=False)
                tab_image.select(lambda: "Image", None, tab_selection)
                tab_video.select(lambda: "Video", None, tab_selection)
            with gr.Accordion(open=True, label="Cropping Options for Source Image or Video"):
                with gr.Row():
                    flag_do_crop_input = gr.Checkbox(value=True, label="do crop (source)")
                    scale = gr.Number(value=2.3, label="source crop scale", minimum=1.8, maximum=3.2, step=0.05)
                    vx_ratio = gr.Number(value=0.0, label="source crop x", minimum=-0.5, maximum=0.5, step=0.01)
                    vy_ratio = gr.Number(value=-0.125, label="source crop y", minimum=-0.5, maximum=0.5, step=0.01)

        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("🎞️ Driving Video") as v_tab_video:
                    with gr.Accordion(open=True, label="Driving Video"):
                        driving_video_input = gr.Video()
                        gr.Examples(
                            examples=[
                                [osp.join(example_video_dir, "d9.mp4")],
                                [osp.join(example_video_dir, "d10.mp4")],
                                [osp.join(example_video_dir, "d11.mp4")],
                                [osp.join(example_video_dir, "d12.mp4")],
                                [osp.join(example_video_dir, "d13.mp4")],
                                [osp.join(example_video_dir, "d14.mp4")],
                            ],
                            inputs=[driving_video_input],
                            cache_examples=False,
                        )
                with gr.TabItem("🖼️ Driving Image") as v_tab_image:
                    with gr.Accordion(open=True, label="Driving Image"):
                        driving_image_input = gr.Image(type="filepath")
                        gr.Examples(
                            examples=[
                                [osp.join(example_portrait_dir, "s9.jpg")],
                                [osp.join(example_portrait_dir, "s6.jpg")],
                                [osp.join(example_portrait_dir, "s10.jpg")],
                                [osp.join(example_portrait_dir, "s5.jpg")],
                                [osp.join(example_portrait_dir, "s7.jpg")],
                                [osp.join(example_portrait_dir, "s12.jpg")],
                            ],
                            inputs=[driving_image_input],
                            cache_examples=False,
                        )

                with gr.TabItem("📁 Driving Pickle") as v_tab_pickle:
                    with gr.Accordion(open=True, label="Driving Pickle"):
                        driving_pickle_input = gr.File(type="filepath", file_types=[".pkl"])
                        gr.Examples(
                            examples=[
                                [osp.join(example_video_dir, "d2.pkl")],
                                [osp.join(example_video_dir, "d8.pkl")],
                            ],
                            inputs=[driving_pickle_input],
                            cache_examples=False,
                        )

                with gr.TabItem("📁 Driving Audio") as v_tab_audio:
                    with gr.Accordion(open=True, label="Driving Audio"):
                        driving_audio_input = gr.Audio(
                            value=None,
                            type="filepath",
                            interactive=True,
                            show_label=False,
                            waveform_options=gr.WaveformOptions(
                                sample_rate=24000,
                            ),
                        )
                        gr.Examples(
                            examples=[
                                [osp.join(example_video_dir, "a-01.wav")],
                            ],
                            inputs=[driving_audio_input],
                            cache_examples=False,
                        )

                v_tab_selection = gr.Textbox(value="Video", visible=False)
                v_tab_video.select(lambda: "Video", None, v_tab_selection)
                v_tab_image.select(lambda: "Image", None, v_tab_selection)
                v_tab_pickle.select(lambda: "Pickle", None, v_tab_selection)
                v_tab_audio.select(lambda: "Audio", None, v_tab_selection)

            # with gr.Accordion(open=False, label="Animation Instructions"):
            # gr.Markdown(load_description("assets/gradio/gradio_description_animation.md"))
            with gr.Accordion(open=True, label="Cropping Options for Driving Video"):
                with gr.Row():
                    flag_crop_driving_video_input = gr.Checkbox(value=False, label="do crop (driving)")
                    scale_crop_driving_video = gr.Number(value=2.2, label="driving crop scale", minimum=1.8,
                                                         maximum=3.2, step=0.05)
                    vx_ratio_crop_driving_video = gr.Number(value=0.0, label="driving crop x", minimum=-0.5,
                                                            maximum=0.5, step=0.01)
                    vy_ratio_crop_driving_video = gr.Number(value=-0.1, label="driving crop y", minimum=-0.5,
                                                            maximum=0.5, step=0.01)

    with gr.Row():
        with gr.Accordion(open=True, label="Animation Options"):
            with gr.Row():
                flag_relative_input = gr.Checkbox(value=True, label="relative motion")
                flag_stitching = gr.Checkbox(value=True, label="stitching")
                driving_multiplier = gr.Number(value=1.0, label="driving multiplier", minimum=0.0, maximum=2.0,
                                               step=0.02)
                cfg_scale = gr.Number(value=4.0, label="cfg_scale", minimum=0.0, maximum=10.0, step=0.5)
                flag_remap_input = gr.Checkbox(value=True, label="paste-back")
                animation_region = gr.Radio(["exp", "pose", "lip", "eyes", "all"], value="all",
                                            label="animation region")
                flag_video_editing_head_rotation = gr.Checkbox(value=False, label="relative head rotation (v2v)")
                driving_smooth_observation_variance = gr.Number(value=1e-7, label="motion smooth strength (v2v)",
                                                                minimum=1e-11, maximum=1e-2, step=1e-8)
                flag_is_animal = gr.Checkbox(value=False, label="is_animal")

    gr.Markdown(load_description("assets/gradio/gradio_description_animate_clear.md"))
    with gr.Row():
        process_button_animation = gr.Button("🚀 Animate", variant="primary")

    with gr.Column():
        with gr.Row():
            with gr.Column():
                output_video_i2v = gr.Video(autoplay=False, label="The animated video in the original image space")
            with gr.Column():
                output_video_concat_i2v = gr.Video(autoplay=False, label="The animated video")
        with gr.Row():
            with gr.Column():
                output_image_i2i = gr.Image(format="png", type="numpy",
                                            label="The animated image in the original image space",
                                            visible=False)
            with gr.Column():
                output_image_concat_i2i = gr.Image(format="png", type="numpy", label="The animated image",
                                                   visible=False)
    with gr.Row():
        process_button_reset = gr.ClearButton(
            [source_image_input, source_video_input, driving_pickle_input, driving_video_input,
             driving_image_input, output_video_i2v, output_video_concat_i2v, output_image_i2i, output_image_concat_i2i],
            value="🧹 Clear")

    # Retargeting
    gr.Markdown(load_description("assets/gradio/gradio_description_retargeting.md"), visible=True)
    with gr.Row(visible=True):
        eye_retargeting_slider.render()
        lip_retargeting_slider.render()
    with gr.Row(visible=True):
        process_button_retargeting = gr.Button("🚗 Retargeting", variant="primary")
        process_button_reset_retargeting = gr.ClearButton(
            [
                eye_retargeting_slider,
                lip_retargeting_slider,
                retargeting_input_image,
                output_image,
                output_image_paste_back
            ],
            value="🧹 Clear"
        )
    with gr.Row(visible=True):
        with gr.Column():
            with gr.Accordion(open=True, label="Retargeting Input"):
                retargeting_input_image.render()
                gr.Examples(
                    examples=[
                        [osp.join(example_portrait_dir, "s9.jpg")],
                        [osp.join(example_portrait_dir, "s6.jpg")],
                        [osp.join(example_portrait_dir, "s10.jpg")],
                        [osp.join(example_portrait_dir, "s5.jpg")],
                        [osp.join(example_portrait_dir, "s7.jpg")],
                        [osp.join(example_portrait_dir, "s12.jpg")],
                    ],
                    inputs=[retargeting_input_image],
                    cache_examples=False,
                )
        with gr.Column():
            with gr.Accordion(open=True, label="Retargeting Result"):
                output_image.render()
        with gr.Column():
            with gr.Accordion(open=True, label="Paste-back Result"):
                output_image_paste_back.render()

    flag_is_animal.change(change_animal_model, inputs=[flag_is_animal])
    # binding functions for buttons
    process_button_retargeting.click(
        # fn=gradio_pipeline.execute_image,
        fn=gpu_wrapped_execute_image,
        inputs=[eye_retargeting_slider, lip_retargeting_slider, retargeting_input_image, flag_do_crop_input],
        outputs=[output_image, output_image_paste_back],
        show_progress=True
    )
    process_button_animation.click(
        fn=gpu_wrapped_execute_video,
        inputs=[
            source_image_input,
            source_video_input,
            driving_video_input,
            driving_image_input,
            driving_pickle_input,
            driving_audio_input,
            flag_relative_input,
            flag_do_crop_input,
            flag_remap_input,
            driving_multiplier,
            flag_stitching,
            flag_crop_driving_video_input,
            flag_video_editing_head_rotation,
            flag_is_animal,
            animation_region,
            scale,
            vx_ratio,
            vy_ratio,
            scale_crop_driving_video,
            vx_ratio_crop_driving_video,
            vy_ratio_crop_driving_video,
            driving_smooth_observation_variance,
            tab_selection,
            v_tab_selection,
            cfg_scale
        ],
        outputs=[output_video_i2v, output_video_i2v, output_video_concat_i2v, output_video_concat_i2v,
                 output_image_i2i, output_image_i2i, output_image_concat_i2i, output_image_concat_i2i],
        show_progress=True
    )

if __name__ == '__main__':
    demo.launch(
        server_port=args.port,
        share=False,
        server_name=args.host_ip
    )
