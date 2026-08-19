import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# CONFIGURAÇÃO DO MEDIAPIPE TASKS (Caminho Automático)
# Descobre a pasta exata onde o main.py está salvo
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
# Junta a pasta atual com o nome do arquivo .task
model_path = os.path.join(diretorio_atual, 'hand_landmarker.task')

# Validação amigável antes de travar o MediaPipe
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"\n\n[ERRO DE ARQUIVO] O arquivo de IA não foi encontrado!\n"
        f"Certifique-se de mover o arquivo 'hand_landmarker.task' para a pasta:\n"
        f"-> {diretorio_atual}\n"
    )

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)

# Cria o detector usando a nova sintaxe
detector = vision.HandLandmarker.create_from_options(options)

# Cria o detector usando a nova sintaxe
detector = vision.HandLandmarker.create_from_options(options)

# Inicializa a câmera usando DirectShow (evita erro de travamento no Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Não foi possível acessar a webcam.")
else:
    print("Rastreamento iniciado com MediaPipe Tasks! Pressione 'q' para fechar.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao receber o quadro da câmera.")
            break
        
        # Espelha o vídeo para o movimento ficar intuitivo
        frame = cv2.flip(frame, 1)
        
        # Converte a imagem de BGR (OpenCV) para RGB (MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Transforma o frame no formato de imagem exigido pelo MediaPipe Tasks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Executa a detecção de mãos
        resultado_deteccao = detector.detect(mp_image)
        
        # Se alguma mão for localizada pelo novo modelo
        if resultado_deteccao.hand_landmarks:
            for pontos_da_mao in resultado_deteccao.hand_landmarks:
                
                alt, larg, _ = frame.shape
                
                # Desenha manualmente as conexões e os pontos estruturais na tela
                for landmark in pontos_da_mao:
                    cx, cy = int(landmark.x * larg), int(landmark.y * alt)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), cv2.FILLED) # Pontos em verde
                
                # Identifica especificamente a ponta do indicador (Índice 8 na estrutura padrão)
                ponta_indicador = pontos_da_mao[8]
                ix, iy = int(ponta_indicador.x * larg), int(ponta_indicador.y * alt)
                
                # Destaca a ponta do indicador colocando uma esfera azul maior
                cv2.circle(frame, (ix, iy), 12, (255, 0, 0), cv2.FILLED)

        # Exibe a tela final processada
        cv2.imshow("Python 3.14 - MediaPipe Tasks", frame)
        
        # Fecha se pressionar a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
