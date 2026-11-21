from backend.app.services.inference import predict_image

detections = predict_image("bus.jpg", confidence=0.25)

print("\nDetections:")
for detection in detections:
    print(detection)

print("\nTotal detections:", len(detections))