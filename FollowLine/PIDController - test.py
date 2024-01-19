from GUI import GUI
from HAL import HAL
import cv2

# Constantes del controlador PID
Kp = 0.08
Ki = 0.005
Kd = 0.005
max_angular_velocity = 0.5
base_linear_velocity = 1.5
min_error_for_high_angular_velocity = 50

# Inicialización de variables
i = 0
prev_error = 0
total_error = 0

while True:
    img = HAL.getImage()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 125, 125), (30, 255, 255))

    try:
        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        M = cv2.moments(contours[0])
    except (IndexError, ZeroDivisionError):
        M = dict()
        M["m00"] = 0

    if M["m00"] != 0:
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
    else:
        cX, cY = 0, 0

    if cX > 0:
        err = 320 - cX

        # Cálculo de los términos PID
        P = Kp * err
        I = Ki * total_error
        D = Kd * (err - prev_error)

        # Calcular la velocidad angular utilizando el controlador PID
        angular_velocity = P + I + D

        # Limitar la velocidad angular máxima
        angular_velocity = max(-max_angular_velocity, min(angular_velocity, max_angular_velocity))

        # Ajustar dinámicamente la velocidad angular en función del error
        if abs(err) > min_error_for_high_angular_velocity:
            angular_velocity *= 1.5  # Aumentar la velocidad angular para curvas pronunciadas

        # Ajustar la velocidad lineal y angular del sistema
        HAL.setV(base_linear_velocity)
        HAL.setW(angular_velocity)

        # Actualizar el error previo y el total del error para la próxima iteración
        prev_error = err
        total_error += err

    GUI.showImage(red_mask)
    print('%d cX: %.2f cY: %.2f Angular Velocity: %.4f' % (i, cX, cY, angular_velocity))
    i += 1
