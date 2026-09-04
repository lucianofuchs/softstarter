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
mensagem_liga = 'T:5s\n'
mensagem_desliga = 'D:5s\n'
mensagem_emergencia = 'E'
tempo = 5
tempoy = 100
tempody = 175
tempo_desliga = 5
tempo_antes = time.perf_counter()
ultima_medicaod = time.perf_counter()
ligado_agora = None
ligado_antes = None
tempo_que_ligou = 0
tempo_ligado = 0
tempo_atual = 0
tempo_que_desligou = 0
ultimo_estado_do_motor = 0
emergencia_acionada = False




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
    num_hands=2
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

        maos_na_emergencia = 0

        if resultado_deteccao.hand_landmarks:
            alt, larg, _ = frame.shape

            for pontos_da_mao in resultado_deteccao.hand_landmarks:
                # Desenha os pontos da mão
                for landmark in pontos_da_mao:
                    cx = int(landmark.x * larg)
                    cy = int(landmark.y * alt)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), cv2.FILLED)

                # Ponta do indicador desta mão
                indicador = pontos_da_mao[8]
                ix = int(indicador.x * larg)
                iy = int(indicador.y * alt)

                cv2.circle(frame, (ix, iy), 12, (255, 0, 0), cv2.FILLED)

                # Área do botão de emergência
                if 425 <= ix <= 575 and 1 <= iy <= 150:
                    maos_na_emergencia += 1

        # Executar somente quando DUAS mãos estiverem no botão
        if maos_na_emergencia >= 2:
            cv2.rectangle(
                frame,
                (425, 1),
                (575, 150),
                (0, 0, 255),
                cv2.FILLED
            )

            dados = mensagem_emergencia.encode("utf-8")
            stm32.write(dados)
            stm32.flush()

            ligado_agora = None
            ultimo_estado_do_motor = 0

        
        if ligado_agora:

            cv2.rectangle(frame,(1,1),(150,150),(0,255,0),cv2.FILLED)
            cv2.putText(frame,'LIGANDO',(35,75),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)
            tempo_atual = time.perf_counter()
            tempo_ligado = tempo_atual - tempo_que_ligou

            if ligado_antes is False or ligado_antes is None:
                tempo_que_ligou = time.perf_counter()
                

            if tempo_atual - tempo_que_ligou >= tempo:
                ligado_agora = None
                tempo_antes = tempo_atual
                tempo_ligado = 0
                ultimo_estado_do_motor = 1

        if ligado_agora is False:
            cv2.rectangle(frame,(1,1),(150,150),(0,0,255),cv2.FILLED)
            cv2.putText(frame,'DESLIGANDO',(15,75),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)
            tempo_atual = time.perf_counter()
            tempo_ligado = tempo_atual - tempo_que_desligou

            if ligado_antes is True or ligado_antes is None:
                tempo_que_desligou = time.perf_counter()

            if tempo_atual - tempo_que_desligou >= tempo_desliga:
                ligado_agora = None
                tempo_antes = tempo_atual
                ultimo_estado_do_motor = 0

        if ligado_agora is None:
            tempo_atual = 0

            if ultimo_estado_do_motor == 1:
                cv2.putText(frame,f'Ligado',(155,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            if ultimo_estado_do_motor == 0:
                cv2.putText(frame,f'Desligado',(155,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

            

        ligado_antes = ligado_agora

        cv2.putText(frame,f'{tempo:.1f}s',(155,75),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,0),2)
        cv2.putText(frame,f'{tempo_desliga:.1f}s',(155,125),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,f'{tempo_ligado:.1f}s',(155,100),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,0),2)
        cv2.circle(frame, (600, tempoy), 8, (255,0,0), cv2.FILLED)
        cv2.circle(frame, (10, tempody), 8, (0,0,255), cv2.FILLED)
        
                

        
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
                cv2.putText(frame,f'{mensagem_desliga}',(ix,iy),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)

                difposy = iy-py
                if difposy < 0: 
                    difposy = abs(difposy)

                difposx = ix-px
                if difposx < 0: 
                    difposx = abs(difposx)

                

                if difposy <= 20 and difposx <= 20:
                    cv2.putText(frame,'tick',(ix,iy),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)

                    if ligado_agora is None:

                        if 570 <= ix <= 630:
                            tempoy = iy
                            if iy < 100:
                                tempoy = 100
                            if iy > 400:
                                tempoy = 400
                            tempo = py - 100
                            tempo = 0.15 * tempo + 5

                            if tempo < 5:
                                tempo = 5
                            if tempo > 55:
                                tempo = 55
                            mensagem_liga = f"T:{tempo:.0f}\n"

                        if 0 <= ix <= 40:
                            tempody = py
                            if py < 175:
                                tempody = 175
                            if py > 450:
                                tempody = 450
                            tempo_desliga = py - 190
                            tempo_desliga = 0.2 * tempo_desliga + 5

                            if tempo_desliga < 5:
                                tempo_desliga = 5
                            if tempo_desliga > 55:
                                tempo_desliga = 55
                            mensagem_desliga = f'D:{tempo_desliga:.0f}\n'                   

                #botao unico
                if iy <= 150 and ix <= 150:
                    if ultimo_estado_do_motor == 1:
                        if ligado_agora is None:
                             ligado_agora = False
                             tempo_atuald = time.perf_counter()
                        if tempo_atuald - ultima_medicaod >= 1.0:
                            dados = mensagem_desliga.encode("utf-8")
                            bytes_enviados = stm32.write(dados)
                            stm32.flush()
                            print("Conteúdo enviado:", dados)
                            print("Quantidade enviada:", bytes_enviados)
                            ultima_medicaod = tempo_atuald
                        
                    if ultimo_estado_do_motor == 0:
                        if ligado_agora is None:

                            tempo_atuald = time.perf_counter()
                            if ligado_agora is None:
                                dados = mensagem_liga.encode("utf-8")
                                bytes_enviados = stm32.write(dados)
                                stm32.flush()
                                print("Conteúdo enviado:", dados)
                                print("Quantidade enviada:", bytes_enviados)
                                ultima_medicaod = tempo_atuald
                                ligado_agora = True

                if 1 <= iy <= 150 and 425 <= ix <= 575:
                    #cv2.rectangle(frame,(425,1),(575,150),(0,0,255),cv2.FILLED)
                    if maos_na_emergencia >= 2:
                        if not emergencia_acionada:
                            dados = mensagem_emergencia.encode('utf-8')
                            bytes_enviados = stm32.write(dados)
                            stm32.flush()
                            print("Conteúdo enviado:", dados)
                            print("Quantidade enviada:", bytes_enviados)
                            emergencia_acionada = True
                            ligado_agora = None
                            ultimo_estado_do_motor = 0
                    else:
                        emergencia_acionada = False
                    
                    
       
        cv2.line(frame,(600,100),(600,400),(255,0,0),3)
        cv2.line(frame,(10,175),(10,450),(0,0,255),3)
        cv2.rectangle(frame,(1,1),(150,150),(100,100,100),3)
        cv2.rectangle(frame,(425,1),(575,150),(0,0,255),3)
        cv2.putText(frame,'EMERGÊNCIA',(432,75),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,0),2)
        #cv2.putText(frame,f'{ultimo_estado_do_motor}',(185,75),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)
        cv2.imshow("Python 3.14 - Softstarter", frame)
        



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
