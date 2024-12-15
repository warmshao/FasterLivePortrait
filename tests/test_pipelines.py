# -*- coding: utf-8 -*-
# @Time    : 2024/12/15
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: test_pipelines.py
import pdb
import pickle
import sys

sys.path.append(".")


def test_joyvasa_pipeline():
    from src.pipelines.joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline

    pipe = JoyVASAAudio2MotionPipeline(
        motion_model_path="checkpoints/JoyVASA/motion_generator/motion_generator_hubert_chinese.pt",
        audio_model_path="checkpoints/chinese-hubert-base",
        motion_template_path="checkpoints/JoyVASA/motion_template/motion_template.pkl")

    audio_path = "assets/examples/driving/a-01.wav"
    motion_data = pipe.gen_motion_sequence(audio_path)
    with open("assets/examples/driving/d1-joyvasa.pkl", "wb") as fw:
        pickle.dump(motion_data, fw)
    pdb.set_trace()


if __name__ == '__main__':
    test_joyvasa_pipeline()
