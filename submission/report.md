# Lab Day 18 — 2D Perception

Mã học viên: **2A202602415**

Link notebook đã chạy: https://github.com/hongcuong26-debug/Track04-Day18-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb

Bản Colab đã thực thi: https://colab.research.google.com/drive/1I0MTnY4Y57XH8Spp4noryFpFoir6dlER

**Trạng thái đồng bộ:** đã hoàn thiện và thực thi trên Colab; đang đưa notebook còn output và hai file kết quả về repo. Link GitHub ở trên chỉ là bản nộp hoàn chỉnh sau khi đồng bộ các file này.

## Thực thi thật trên Colab

Đã thực hiện `Runtime → Restart session and run all` trên Tesla T4, không có lỗi Python trong 94 ô của bài lab. Môi trường: Python 3.13.15, torch 2.11.0+cu130, torchvision 0.26.0+cu130, ultralytics 8.4.171; `device = cuda`. Huấn luyện tiger-pose đủ **40 epoch, imgsz=640, batch=16**, seed=0; lần chạy cuối mất **5.8 phút**.

| Chỉ số trên 53 ảnh val | Giá trị |
|---|---:|
| Box mAP50–95 | 0.930349 |
| Pose mAP50 | 0.995000 |
| Pose mAP50–95 | 0.457346 |
| OKS trung bình | 0.745000 |
| Không phát hiện được hổ | 0/53 |
| Nghi đảo trái/phải theo phép đổi nhãn kiểm tra | 4/53 |

mAP50 cao không đồng nghĩa keypoint định vị chính xác: mAP ở các ngưỡng OKS chặt hơn giảm rõ, còn lỗi tập trung ở bàn chân khi các chân bước/che nhau. Q11 phân tích cụ thể `Frame_31.jpg` và `Frame_51.jpg`, kèm sai số từng khớp và cách cải thiện.

## Latency lần chạy cuối

Trung bình 30 lần trên T4, theo `Results.speed`, đơn vị ms.

| Cấu hình | Preprocess | Inference | Postprocess | Số box |
|---|---:|---:|---:|---:|
| One-to-many + NMS, conf 0.25 | 1.82 | 9.83 | 1.30 | 5 |
| One-to-many + NMS, conf 0.001 | 1.73 | 9.01 | 1.32 | 203 |
| One-to-one, conf 0.25 | 1.76 | 9.33 | 0.43 | 5 |
| One-to-one, conf 0.001 | 1.78 | 9.80 | 0.45 | 204 |

Hai mức conf đều giảm 0.87 ms postprocess trong lần đo này. Kết quả được ghi đúng như quan sát, không giả định conf 0.001 luôn có chênh lệch lớn nhất.

## Segmentation và keypoints

- Semantic có 8 vùng person, trong khi ảnh có 4 người và Mask R-CNN tìm được 4 người; 63% pixel trong box bus bị gán `train`.
- Hungarian matching có 5 cặp mask, IoU [0.939, 0.842, 0.932, 0.922, 0.890].
- SAM tạo 5 object; IoU polygon lưu trong `bus.txt` với mask SAM gốc là [0.970, 0.982, 0.967, 0.978, 0.983]; `check_autolabel` đạt.
- Prompt A: 47441 pixel, IoU 0.99; prompt B: 2714 pixel, IoU làm tròn 0.00.
- Đã chạy `KP_THR=-100` và quan sát các khớp chân ngoài ảnh được dự đoán sát đáy khung.
- Pose trên bus gốc trả 5 box dù thực tế có 4 người; ảnh xoay trả 3 box, thân nghiêng 90°, 94°, 83°.
- `FLIP_IDX = [0, 1, 2, 3, 7, 6, 5, 4, 10, 11, 8, 9]` được kiểm tra theo tên trong YAML tiger-pose của đúng package.

## Kiểm tra và bonus

10 deliverable local đạt, gồm 38 phép thử gốc và năm gate; không dùng phao. Checklist `final_report()` trên Colab đạt toàn bộ mục bắt buộc, 12/12 câu hỏi đã hoàn thiện, không còn `<<ĐIỀN SỐ TỪ OUTPUT>>`. `ket_qua.json` và `autolabel/bus.txt` do các ô gốc sinh ra, không sửa số liệu bằng tay.

Bonus 1D đã làm: `average_precision` đạt và có hình PR. Không thực hiện bonus 4C hoặc bài tập về nhà ONNX/auto-label thêm ảnh, nên không yêu cầu điểm cho hai phần này.

Chạy kiểm tra local: `.venv\Scripts\python.exe tools\run_checks.py`.

Sau khi đủ file xuất từ Colab, chạy kiểm tra bản nộp: `.venv\Scripts\python.exe tools\validate_submission.py`.

Chỉ nộp URL repo GitHub public lên VLearn, không mở PR; giữ repo public đến khi có điểm.
