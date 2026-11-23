
from backend.app.services.inference import predict_video

video_path = "test_video.mp4"
output_path = "test_video_annotated.mp4"

print("Processing video...")

result = predict_video(
    video_path=video_path,
    output_path=output_path,
    confidence=0.25
)

print("\nVideo processing completed!")
print("Processed frames:", result["processed_frames"])
print("Detection counts:", result["detection_counts"])
print("Output video:", result["output_path"])
