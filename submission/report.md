# Lab Day 18 — 2D Perception

Mã học viên: **2A202602415**

Link notebook đã chạy: https://github.com/hongcuong26-debug/Track04-Day18-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb

**Trạng thái hiện tại:** notebook đã điền code và khung Q1–Q12; chưa chạy toàn bộ trên Colab T4. Link trên là đường dẫn dự kiến của bản nộp, chưa xác nhận bản online đã có output. Chưa có số liệu latency, mask SAM hoặc mAP từ lần chạy thật.

## Phần đã chuẩn bị

- Tự viết `box_iou`, `nms`, `batched_nms`, `mask_iou`, `polygon_to_mask`, `mask_to_yolo_seg`, `oks`, `joint_angle`.
- Bonus 1D: hoàn thành `average_precision`; kiểm tra và hình PR sẽ chạy trong notebook.
- `FLIP_IDX = [0, 1, 2, 3, 7, 6, 5, 4, 10, 11, 8, 9]`, suy ra từ tên keypoint trong [YAML tiger-pose của Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/tiger-pose.yaml). Các cặp là hock sau 4↔7, paw sau 5↔6, wrist trước 8↔10, paw trước 9↔11; bốn điểm trục giữa giữ nguyên.
- Không đổi bộ kiểm tra, gate, metadata, thứ tự ô hoặc output có sẵn; không sử dụng phao.
- Khung Q2, Q4–Q9 và Q11 có `<<ĐIỀN SỐ TỪ OUTPUT>>` để ghi kết quả/quan sát thực tế. Không dùng số minh họa trong README thay số đo của mình.

## Chạy và hoàn thiện trên Colab T4

1. Upload `lab_2d_perception_student.ipynb` từ repo local lên Colab, chọn T4 GPU, rồi **Restart session and run all**. Phần 0 phải in `device = cuda` và đủ chín dấu tải thành công.
2. Phần 3A: tạm đổi `KP_THR = -100`, chạy lại ô vẽ, quan sát đầu gối/mắt cá và hoàn thiện Q7; đưa về `2.0` trước lần chạy toàn bộ cuối.
3. Phần 4: bảo đảm ô ghi `tiger-pose-anat.yaml` chạy sau ô `FLIP_IDX` và trước train; train đủ **40 epoch, imgsz=640**. Nếu hết bộ nhớ, đổi `batch=16` thành `batch=8`.
4. Ghi đủ bốn cấu hình latency vào Q2; ghi các quan sát ở Q4–Q6; đọc đồ thị OKS ở Q8 và số người ở Q9. Q11 phải nêu **hai kiểu lỗi thật**, mỗi kiểu có tên ảnh trong sáu ảnh tệ nhất, keypoint/vùng cơ thể, biểu hiện sai và cách sửa cụ thể.
5. Thay mọi `<<ĐIỀN SỐ TỪ OUTPUT>>`, chạy lại các ô trả lời, rồi chạy toàn bộ một lần cuối để lưu output nhất quán. Chạy `final_report()` và kiểm tra mọi dòng bắt buộc ✅; kiểm tra riêng train đủ 40 epoch vì checklist chỉ kiểm tra sự hiện diện bảng mAP.
6. Tải notebook **còn output** và `submission.zip`; giải nén sao cho repo có `submission/ket_qua.json` và `submission/autolabel/bus.txt`. Giữ lại báo cáo này; không lồng `submission/submission/`. Chỉ `final_report()` được sinh `ket_qua.json`, không điền bằng tay.
7. Cập nhật trạng thái báo cáo và link nếu repo/branch thay đổi, đẩy các file lên GitHub public, kiểm tra truy cập khi chưa đăng nhập, rồi nộp URL repo lên VLearn. Không mở PR.

## Kiểm tra local

PowerShell: `.venv\Scripts\python.exe tools\run_checks.py`.

Script dùng bộ kiểm tra gốc, dữ liệu tổng hợp và YAML của đúng package `ultralytics==8.4.171`, không tải dataset/weights và không tạo `ket_qua.json`. Nó kiểm tra 10 deliverable gồm bonus AP và FLIP_IDX, các ca biên, năm gate, cú pháp Python và schema notebook.

Kết quả ngày 06/10/2026: **10/10 deliverable đạt, 38 phép thử gốc đạt, năm gate đạt**, các ca biên và kiểm tra schema/cú pháp đạt. Môi trường CPU: Python 3.11.16, torch 2.14.1+cpu, torchvision 0.29.1+cpu, ultralytics 8.4.171. Kết quả này xác nhận các hàm tự viết; không thay thế output inference hoặc train T4 cần nộp.

## Bonus 4C (tùy chọn, chưa chạy)

Nếu làm thêm, đặt `RUN_4C = True` và `TRAIN_IDENTITY = True` ở ô 91; cần thêm một lần train. Ghi bảng hai model trên val gốc/val gương và số liệu thật tại đây sau khi chạy.

mAP trên val chỉ có hổ quay phải có thể che lỗi `flip_idx` vì không kiểm tra hướng đối xứng và quy ước giải phẫu ở tư thế đó. Val lật gương với nhãn hoán đổi đúng và val thực tế gồm cả hai hướng giúp kiểm tra khả năng tổng quát hóa; nên phân tích thêm sai số từng cặp chân thay vì chỉ nhìn một mAP tổng hợp.

Chưa thực hiện bài tập về nhà ONNX hoặc auto-label 20–30 ảnh; không tính điểm bonus cho phần chưa chạy.
