import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import serial
import time

stm32 = serial.Serial(
    port="COM3",
    baudrate=115200,
    timeout=1
)

time.sleep(2)
mensagem = ''
tempo = 0
tempoy = 100
ultima_medicao = time.perf_counter()

diretorio_atual = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(diretorio_atual, 'hand_landmarker.task')

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

detector = vision.HandLandmarker.create_from_options(options)

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
        
        frame = cv2.flip(frame, 1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        resultado_deteccao = detector.detect(mp_image)
        
        if resultado_deteccao.hand_landmarks:
            for pontos_da_mao in resultado_deteccao.hand_landmarks:
                
                alt, larg, _ = frame.shape
                
                for landmark in pontos_da_mao:
                    cx, cy = int(landmark.x * larg), int(landmark.y * alt)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), cv2.FILLED)
                
                ponta_indicador = pontos_da_mao[8]
                ix, iy = int(ponta_indicador.x * larg), int(ponta_indicador.y * alt)

                ponta_polegar = pontos_da_mao[4]
                px, py = int(ponta_polegar.x * larg), int(ponta_polegar.y * alt)
                
                cv2.circle(frame, (ix, iy), 12, (255, 0, 0), cv2.FILLED)
                cv2.circle(frame, (px, py), 12, (0, 100, 180), cv2.FILLED)

                difposy = iy-py
                if difposy < 0: 
                    difposy = abs(difposy)

                difposx = ix-px
                if difposx < 0: 
                    difposx = abs(difposx)

                    

                if difposy <= 20 and difposx <= 20:
                    cv2.putText(frame,'tick',(ix,iy),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)

                    if 570 <= ix <= 630:
                        tempoy = iy
                        tempo = py - 100
                        tempo = 0.15 * tempo + 5
                        mensagem = f"T:{tempo:.0f}\n"
                    
                cv2.circle(frame, (600, tempoy), 8, (255,0,0), cv2.FILLED)
                cv2.putText(frame,f'{tempo:.1f}s',(5,200),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,0),2)

                if iy <= 150 and ix <= 150:
                    cv2.rectangle(frame,(1,1),(150,150),(0,255,0),cv2.FILLED)

                    tempo_atual = time.perf_counter()
                    if tempo_atual - ultima_medicao >= 1.0:
                        dados = mensagem.encode("utf-8")

                        bytes_enviados = stm32.write(dados)
                        stm32.flush()

                        print("Conteúdo enviado:", dados)
                        print("Quantidade enviada:", bytes_enviados)
                        
                        ultima_medicao = tempo_atual
                    

                
        cv2.line(frame,(600,100),(600,400),(255,0,0),3)
        cv2.rectangle(frame,(1,1),(150,150),(0,255,0),3)
        cv2.rectangle(frame,(153,1),(303,150),(0,0,255),3)
        cv2.putText(frame,'START',(35,75),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)
        cv2.putText(frame,'STOP',(185,75),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)
        cv2.imshow("Python 3.14 - Softstarter", frame)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
