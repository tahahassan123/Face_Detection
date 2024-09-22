import sys
import cv2
import pygame
import cv2 as cv
import imutils

from ultralytics import YOLO
from tkinter import filedialog

pygame.init()
pygame.font.init()

video = ()
trajectory = []
predicted_trajectory = []

skyblue = (0, 128, 255)
lightgrey = (218, 208, 208)
nightblue = (0, 25, 51)
roadgrey = (64, 64, 64)
red = (255, 0, 0)
lightblue = (115, 202, 246)
white = (255, 255, 255)
transparent = (0, 0, 0, 0)
black = (0, 0, 0)

imageWidth = 250
imageHeight = 250

WIDTH = 720
HEIGHT = 500
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))

button = pygame.image.load("images/button.png").convert_alpha()
pygame.display.set_caption("Face Detection", "FD")

button1 = pygame.transform.scale(button, (350, 80))
button2 = pygame.transform.scale(button, (380, 110))
font_ubuntu = pygame.font.SysFont("ubuntu", 30)
font_ubuntu2 = pygame.font.SysFont("ubuntu", 18)
menu = True


def draw_bounding_boxes(results, frame):
    for result in results:
        boxes = result.boxes.cpu().numpy()
        xyxys = boxes.xyxy

        for xyxy in xyxys:
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)

    return frame


def detect_ball(video_path):
    global menu
    cap = cv2.VideoCapture(video_path)
    model = YOLO("runs/detect/train/weights/best.pt")
    frame_count = 0
    original_width, original_height, fps = (int(cap.get(x)) for x in
                                            (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    while True:
        ret, frame = cap.read()
        if ret is False:
            break

        results = model.predict(frame, conf=0.4)
        # results = model.track(frame, persist=True, show=True, tracker="botsort.yaml")

        frame = draw_bounding_boxes(results, frame)
        resized_frame = imutils.resize(frame, height=imageHeight, width=imageWidth)

        cv2.imshow("Face Detection", resized_frame)

        key = cv2.waitKey(1)
        if key == 27:
            menu = False
            break

        if cv.getWindowProperty('Face Detection', cv.WND_PROP_VISIBLE) < 1:
            menu = False
            break

    # release the cap object
    cap.release()
    # close all windows
    cv2.destroyAllWindows()


def app_ui():
    global video
    while menu:
        WINDOW.fill(skyblue)
        mx, my = pygame.mouse.get_pos()

        selected_file = font_ubuntu2.render("Selected File: " + str(video), True, black)
        select = font_ubuntu.render("Select Video", True, black)
        play_video = font_ubuntu.render("Play Video", True, black)
        quit_menu = font_ubuntu.render("Quit", True, black)
        error = font_ubuntu.render("Error: No file selected", True, red)

        WINDOW.blit(button1, (180, 100))
        WINDOW.blit(button1, (180, 200))
        WINDOW.blit(button1, (180, 300))

        WINDOW.blit(selected_file, (50, 30))
        WINDOW.blit(select, (270, 118))
        WINDOW.blit(play_video, (280, 218))
        WINDOW.blit(quit_menu, (318, 318))

        if (510 > mx > 200) and (170 > my > 105):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
            WINDOW.blit(button2, (165, 88))
            WINDOW.blit(select, (270, 118))
        elif (510 > mx > 200) and (270 > my > 205):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
            WINDOW.blit(button2, (165, 188))
            WINDOW.blit(play_video, (280, 218))
        elif (510 > mx > 200) and (370 > my > 305):
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
            WINDOW.blit(button2, (165, 288))
            WINDOW.blit(quit_menu, (318, 318))
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if (510 > mx > 200) and (170 > my > 105):
                    video = filedialog.askopenfilename()
                    break

                elif (510 > mx > 200) and (270 > my > 205):

                    if video != ():
                        detect_ball(video)
                    else:
                        WINDOW.blit(error, (250, 18))
                        pygame.display.update()

                elif (510 > mx > 200) and (370 > my > 305):
                    pygame.quit()
                    sys.exit()

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


app_ui()
