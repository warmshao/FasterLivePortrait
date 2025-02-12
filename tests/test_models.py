# -*- coding: utf-8 -*-
# @Time    : 2024/7/13 17:20
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: test_models.py
import json
import os, sys
import pdb

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def test_warping_spade_model():
    """
    test warping model in onnx and trt
    :return:
    """
    import numpy as np
    import time
    from src.models import WarpingSpadeModel

    # tensorrt 模型加载
    trt_kwargs = dict(
        predict_type="trt",
        model_path="./checkpoints/liveportrait_animal_onnx/warping_spade-fix.trt",
    )

    trt_model = WarpingSpadeModel(**trt_kwargs)

    # onnx 模型加载
    onnx_kwargs = dict(
        predict_type="ort",
        model_path="./checkpoints/liveportrait_animal_onnx/warping_spade.onnx",
    )
    onnx_model = WarpingSpadeModel(**onnx_kwargs)

    feature_3d = np.random.randn(1, 32, 16, 64, 64)
    kp_source = np.random.randn(1, 21, 3)
    kp_driving = np.random.randn(1, 21, 3)

    trt_rets = trt_model.predict(feature_3d, kp_source, kp_driving)
    onnx_rets = onnx_model.predict(feature_3d, kp_source, kp_driving)

    # for i in range(len(trt_rets)):
    print(f"output max diff:{np.abs(trt_rets - onnx_rets).max()}")
    infer_times = []
    for _ in range(30):
        t0 = time.time()
        trt_rets = trt_model.predict(feature_3d, kp_source, kp_driving)
        infer_times.append(time.time() - t0)
    print(
        "{} tensorrt inference time: min: {}, max: {}, mean: {}".format(WarpingSpadeModel.__name__, np.min(infer_times),
                                                                        np.max(infer_times), np.median(infer_times)))

    infer_times = []
    for _ in range(30):
        t0 = time.time()
        onnx_rets = onnx_model.predict(feature_3d, kp_source, kp_driving)
        infer_times.append(time.time() - t0)
    print("{} onnx inference time: min: {}, max: {}, mean: {}".format(WarpingSpadeModel.__name__, np.min(infer_times),
                                                                      np.max(infer_times), np.median(infer_times)))


def test_motion_extractor_model():
    """
    test motion_extractor model in onnx and trt
    :return:
    """
    import numpy as np
    import time
    import cv2
    from src.models import MotionExtractorModel

    # tensorrt 模型加载
    trt_kwargs = dict(
        predict_type="trt",
        model_path="./checkpoints/liveportrait_animal_onnx/motion_extractor.trt",
        debug=True
    )

    trt_model = MotionExtractorModel(**trt_kwargs)

    # onnx 模型加载
    onnx_kwargs = dict(
        predict_type="ort",
        model_path="./checkpoints/liveportrait_animal_onnx/motion_extractor.onnx",
        debug=True
    )
    onnx_model = MotionExtractorModel(**onnx_kwargs)

    img_bgr = cv2.imread("assets/examples/source/s1.jpg")
    img_rgb = img_bgr[:, :, ::-1]
    input = cv2.resize(img_rgb, (256, 256))

    trt_rets = trt_model.predict(input)
    onnx_rets = onnx_model.predict(input)
    for i in range(len(trt_rets)):
        print(f"output {i} max diff:{np.abs(trt_rets[i] - onnx_rets[i]).max()}")
    pdb.set_trace()
    infer_times = []
    for _ in range(30):
        t0 = time.time()
        trt_rets = trt_model.predict(input)
        infer_times.append(time.time() - t0)
    print("{} tensorrt inference time: min: {}, max: {}, mean: {}".format(MotionExtractorModel.__name__,
                                                                          np.min(infer_times),
                                                                          np.max(infer_times), np.median(infer_times)))

    infer_times = []
    for _ in range(30):
        t0 = time.time()
        onnx_rets = onnx_model.predict(input)
        infer_times.append(time.time() - t0)
    print(
        "{} onnx inference time: min: {}, max: {}, mean: {}".format(MotionExtractorModel.__name__, np.min(infer_times),
                                                                    np.max(infer_times), np.median(infer_times)))


def test_appearance_extractor_model():
    """
    test motion_extractor model in onnx and trt
    :return:
    """
    import numpy as np
    import time
    import cv2
    from src.models import AppearanceFeatureExtractorModel

    # tensorrt 模型加载
    trt_kwargs = dict(
        predict_type="trt",
        model_path="./checkpoints/liveportrait_onnx/appearance_feature_extractor.trt",
    )

    trt_model = AppearanceFeatureExtractorModel(**trt_kwargs)

    # onnx 模型加载
    onnx_kwargs = dict(
        predict_type="ort",
        model_path="./checkpoints/liveportrait_onnx/appearance_feature_extractor.onnx",
    )
    onnx_model = AppearanceFeatureExtractorModel(**onnx_kwargs)

    img_bgr = cv2.imread("assets/examples/source/s1.jpg")
    img_rgb = img_bgr[:, :, ::-1]
    input = cv2.resize(img_rgb, (256, 256))

    trt_rets = trt_model.predict(input)
    onnx_rets = onnx_model.predict(input)
    print(f"output max diff:{np.abs(trt_rets - onnx_rets).max()}")
    pdb.set_trace()
    infer_times = []
    for _ in range(20):
        t0 = time.time()
        trt_rets = trt_model.predict(input)
        infer_times.append(time.time() - t0)
    print("{} tensorrt inference time: min: {}, max: {}, mean: {}".format(AppearanceFeatureExtractorModel.__name__,
                                                                          np.min(infer_times),
                                                                          np.max(infer_times), np.mean(infer_times)))

    # onnx is so slow, don't why, maybe the grid_sample op not implemented well?
    infer_times = []
    for _ in range(20):
        t0 = time.time()
        onnx_rets = onnx_model.predict(input)
        infer_times.append(time.time() - t0)
    print(
        "{} onnx inference time: min: {}, max: {}, mean: {}".format(AppearanceFeatureExtractorModel.__name__,
                                                                    np.min(infer_times),
                                                                    np.max(infer_times), np.mean(infer_times)))


def test_landmark_model():
    """
    test motion_extractor model in onnx and trt
    :return:
    """
    import numpy as np
    import time
    import cv2
    from src.models import LandmarkModel

    # tensorrt 模型加载
    trt_kwargs = dict(
        predict_type="trt",
        model_path="./checkpoints/liveportrait_onnx/landmark.trt",
        debug=True
    )

    trt_model = LandmarkModel(**trt_kwargs)

    # onnx 模型加载
    onnx_kwargs = dict(
        predict_type="ort",
        model_path="./checkpoints/liveportrait_onnx/landmark.onnx",
        debug=True
    )
    onnx_model = LandmarkModel(**onnx_kwargs)

    img_bgr = cv2.imread("assets/examples/source/s1.jpg")
    img_rgb = img_bgr[:, :, ::-1]
    input = cv2.resize(img_rgb, (224, 224))

    trt_rets = trt_model.predict(input)
    onnx_rets = onnx_model.predict(input)
    print(f"output max diff:{np.abs(trt_rets - onnx_rets).max()}")
    pdb.set_trace()

    infer_times = []
    for _ in range(30):
        t0 = time.time()
        trt_rets = trt_model.predict(input)
        infer_times.append(time.time() - t0)
    print("{} tensorrt inference time: min: {}, max: {}, mean: {}".format(LandmarkModel.__name__,
                                                                          np.min(infer_times),
                                                                          np.max(infer_times), np.median(infer_times)))

    # onnx is so slow, don't why, maybe the grid_sample op not implemented well?
    infer_times = []
    for _ in range(30):
        t0 = time.time()
        onnx_rets = onnx_model.predict(input)
        infer_times.append(time.time() - t0)
    print(
        "{} onnx inference time: min: {}, max: {}, mean: {}".format(LandmarkModel.__name__,
                                                                    np.min(infer_times),
                                                                    np.max(infer_times), np.median(infer_times)))


def test_face_analysis_model():
    import numpy as np
    import cv2
    import time
    from src.models import FaceAnalysisModel
    img_bgr = cv2.imread("assets/examples/source/s1.jpg")

    # onnx 模型加载
    onnx_kwargs = dict(
        predict_type="ort",
        model_path=["./checkpoints/liveportrait_onnx/retinaface_det_static.onnx",
                    "./checkpoints/liveportrait_onnx/face_2dpose_106_static.onnx"],
    )
    onnx_model = FaceAnalysisModel(**onnx_kwargs)

    # tensorrt 模型加载
    trt_kwargs = dict(
        predict_type="trt",
        model_path=["./checkpoints/liveportrait_onnx/retinaface_det_static.trt",
                    "./checkpoints/liveportrait_onnx/face_2dpose_106_static.trt"],
    )

    trt_model = FaceAnalysisModel(**trt_kwargs)

    trt_rets = trt_model.predict(img_bgr)[0]
    onnx_rets = onnx_model.predict(img_bgr)[0]
    for key in trt_rets:
        print(f"output {key} max diff:{np.abs(trt_rets[key] - onnx_rets[key]).max()}")
    pdb.set_trace()
    infer_times = []
    for _ in range(30):
        t0 = time.time()
        trt_rets = trt_model.predict(img_bgr)
        infer_times.append(time.time() - t0)
    print("{} tensorrt inference time: min: {}, max: {}, mean: {}".format(FaceAnalysisModel.__name__,
                                                                          np.min(infer_times),
                                                                          np.max(infer_times), np.median(infer_times)))

    infer_times = []
    for _ in range(30):
        t0 = time.time()
        onnx_rets = onnx_model.predict(img_bgr)
        infer_times.append(time.time() - t0)
    print(
        "{} onnx inference time: min: {}, max: {}, mean: {}".format(FaceAnalysisModel.__name__, np.min(infer_times),
                                                                    np.max(infer_times), np.median(infer_times)))


def test_stitching_model():
    """
    test stitching model in onnx and trt
    :return:
    """
    import numpy as np
    import time
    from src.models import StitchingModel

    # tensorrt 模型加载
    trt_kwargs = dict(
        predict_type="trt",
        model_path="./checkpoints/liveportrait_onnx/stitching.trt",
    )

    trt_model = StitchingModel(**trt_kwargs)

    # onnx 模型加载
    onnx_kwargs = dict(
        predict_type="ort",
        model_path="./checkpoints/liveportrait_onnx/stitching.onnx"
    )
    onnx_model = StitchingModel(**onnx_kwargs)

    input = np.random.randn(1, 126)

    trt_rets = trt_model.predict(input)
    onnx_rets = onnx_model.predict(input)
    print(f"output max diff:{np.abs(trt_rets - onnx_rets).max()}")

    infer_times = []
    for _ in range(20):
        t0 = time.time()
        trt_rets = trt_model.predict(input)
        infer_times.append(time.time() - t0)
    print("{} tensorrt inference time: min: {}, max: {}, mean: {}".format(StitchingModel.__name__,
                                                                          np.min(infer_times),
                                                                          np.max(infer_times), np.median(infer_times)))

    # onnx is so slow, don't why, maybe the grid_sample op not implemented well?
    infer_times = []
    for _ in range(20):
        t0 = time.time()
        onnx_rets = onnx_model.predict(input)
        infer_times.append(time.time() - t0)
    print(
        "{} onnx inference time: min: {}, max: {}, mean: {}".format(StitchingModel.__name__,
                                                                    np.min(infer_times),
                                                                    np.max(infer_times), np.median(infer_times)))


def test_mediapipe_face():
    img_path = ""
    import cv2
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    os.makedirs('./results/mediapipe_test', exist_ok=True)
    # For static images:
    IMAGE_FILES = ["assets/examples/source/s9.jpg"]
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for landmark in face_landmarks.landmark:
                    # 提取每个关键点的 x, y, z 坐标
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                pdb.set_trace()
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
            cv2.imwrite('./results/mediapipe_test/' + os.path.basename(file), annotated_image)


def test_kokoro_model():
    import os
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    import torchaudio

    from src.models.kokoro.models import build_model
    from src.models.kokoro.kokoro import generate
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL = build_model('checkpoints/Kokoro-82M/kokoro-v0_19.pth', device)
    VOICE_NAME = [
        'af',  # Default voice is a 50-50 mix of Bella & Sarah
        'af_bella', 'af_sarah', 'am_adam', 'am_michael',
        'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
        'af_nicole', 'af_sky',
    ][0]
    VOICEPACK = torch.load(f'checkpoints/Kokoro-82M/voices/{VOICE_NAME}.pt', weights_only=True).to(device)
    print(f'Loaded voice: {VOICE_NAME}')

    text = "How could I know? It's an unanswerable question. Like asking an unborn child if they'll lead a good life. They haven't even been born."
    audio, out_ps = generate(MODEL, text, VOICEPACK, lang=VOICE_NAME[0])
    audio_save_path = "./results/kokoro-82m/kokoro_test.wav"
    os.makedirs(os.path.dirname(audio_save_path), exist_ok=True)
    torchaudio.save(audio_save_path, audio[0], 24000)
    print(f"audio save to {audio_save_path}")


def test_kokoro_v1_model():
    # import os
    # os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    # os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    import torchaudio
    from kokoro import KPipeline, KModel
    import soundfile as sf
    import numpy as np
    import torch

    # 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
    # 🇯🇵 'j' => Japanese: pip install misaki[ja]
    # 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
    voice = 'jf_tebukuro'
    with open("checkpoints/Kokoro-82M/config.json", "r", encoding="utf-8") as fin:
        model_config = json.load(fin)
    model = KModel(config=model_config, model="checkpoints/Kokoro-82M/kokoro-v1_0.pth")
    pipeline = KPipeline(lang_code=voice[0], model=model)  # <= make sure lang_code matches voice
    model.voices = {}
    voice_path = "checkpoints/Kokoro-82M/voices"
    for vname in os.listdir(voice_path):
        pipeline.voices[os.path.splitext(vname)[0]] = torch.load(os.path.join(voice_path, vname), weights_only=True)
    # This text is for demonstration purposes only, unseen during training
    # text = '''
    # The sky above the port was the color of television, tuned to a dead channel.
    # "It's not like I'm using," Case heard someone say, as he shouldered his way through the crowd around the door of the Chat. "It's like my body's developed this massive drug deficiency."
    # It was a Sprawl voice and a Sprawl joke. The Chatsubo was a bar for professional expatriates; you could drink there for a week and never hear two words in Japanese.
    #
    # These were to have an enormous impact, not only because they were associated with Constantine, but also because, as in so many other areas, the decisions taken by Constantine (or in his name) were to have great significance for centuries to come. One of the main issues was the shape that Christian churches were to take, since there was not, apparently, a tradition of monumental church buildings when Constantine decided to help the Christian church build a series of truly spectacular structures. The main form that these churches took was that of the basilica, a multipurpose rectangular structure, based ultimately on the earlier Greek stoa, which could be found in most of the great cities of the empire. Christianity, unlike classical polytheism, needed a large interior space for the celebration of its religious services, and the basilica aptly filled that need. We naturally do not know the degree to which the emperor was involved in the design of new churches, but it is tempting to connect this with the secular basilica that Constantine completed in the Roman forum (the so-called Basilica of Maxentius) and the one he probably built in Trier, in connection with his residence in the city at a time when he was still caesar.
    #
    # [Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
    # '''
    text = '「もしおれがただ偶然、そしてこうしようというつもりでなくここに立っているのなら、ちょっとばかり絶望するところだな」と、そんなことが彼の頭に思い浮かんだ。'
    # text = '中國人民不信邪也不怕邪，不惹事也不怕事，任何外國不要指望我們會拿自己的核心利益做交易，不要指望我們會吞下損害我國主權、安全、發展利益的苦果！'
    # text = 'Los partidos políticos tradicionales compiten con los populismos y los movimientos asamblearios.'
    # text = 'Le dromadaire resplendissant déambulait tranquillement dans les méandres en mastiquant de petites feuilles vernissées.'
    # text = 'ट्रांसपोर्टरों की हड़ताल लगातार पांचवें दिन जारी, दिसंबर से इलेक्ट्रॉनिक टोल कलेक्शनल सिस्टम'
    # text = "Allora cominciava l'insonnia, o un dormiveglia peggiore dell'insonnia, che talvolta assumeva i caratteri dell'incubo."
    # text = 'Elabora relatórios de acompanhamento cronológico para as diferentes unidades do Departamento que propõem contratos.'

    # 4️⃣ Generate, display, and save audio files in a loop.
    generator = pipeline(
        text, voice=voice,  # <= change voice here
        speed=1, split_pattern=r'\n+'
    )
    audios = []
    for i, (gs, ps, audio) in enumerate(generator):
        audios.append(audio)
    audios = np.concatenate(audios)
    sf.write(f'./results/kokoro-82m/kokoro_v1_0_{voice}.wav', audios, 24000)  # save each audio file
    print(f'./results/kokoro-82m/kokoro_v1_0_{voice}.wav')


if __name__ == '__main__':
    # test_warping_spade_model()
    # test_motion_extractor_model()
    # test_landmark_model()
    # test_face_analysis_model()
    # test_appearance_extractor_model()
    # test_stitching_model()
    # test_mediapipe_face()
    # test_kokoro_model()
    test_kokoro_v1_model()
