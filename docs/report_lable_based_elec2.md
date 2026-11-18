# Báo cáo thực hành — Label-based Drift Detection trên Elec2

**Người thực hiện :** Võ Trọng Nghĩa

**Dự án:** Adaptive Model Updating in Automated Data Pipelines  
**Mục tiêu task:** Hiểu và kiểm chứng quy trình phát hiện thay đổi trên **luồng lỗi có nhãn**, rồi so sánh mô hình giữ nguyên với mô hình huấn luyện lại khi được cảnh báo.  
**Nguồn kết quả:** notebook `Elec2_Label_Based_Drift_Detection.ipynb` đã chạy, Python 3.11.11, River 0.23.0.

## 1. Cần hiểu gì sau task này?

Task kiểm tra một chuỗi xử lý: **dữ liệu đến → dự đoán → nhận nhãn thật → tính lỗi → detector phân tích lỗi → (nếu cần) huấn luyện lại → tiếp tục dự đoán**.

Điều quan trọng là phân biệt **phát hiện thay đổi** và **cập nhật mô hình**: ADWIN/DDM chỉ đưa ra tín hiệu; phần điều khiển mới quyết định huấn luyện lại. Cảnh báo trên lỗi **không tự chứng minh** sự tồn tại của concept drift.

## 2. Dữ liệu và cách đánh giá

- **Elec2:** 45.312 quan sát liên tiếp, 8 đặc trưng, nhãn nhị phân thể hiện biến động giá điện.
- **Prequential (test-then-train):** tại mẫu thứ `t`, mô hình dự đoán **trước** khi dùng nhãn của mẫu này để học. Sau đó tính lỗi `e_t = 0` nếu dự đoán đúng, `e_t = 1` nếu sai.
- **Error stream:** chuỗi `0/1` theo thời gian, chính là dữ liệu đầu vào của ADWIN và DDM trong task này.
- **No-change baseline:** luôn dự đoán nhãn hiện tại bằng nhãn vừa quan sát ở bước trước; không cần ML.

Trong thí nghiệm so sánh chính, **2.000 mẫu đầu** được dùng làm warm-up và **43.312 mẫu còn lại** dùng để đánh giá cùng ba chiến lược. Biểu đồ accuracy dùng cửa sổ trượt **500 dự đoán**; đây là độ chính xác cục bộ, không phải accuracy tích lũy.

## 3. Học các thuật toán theo đúng luồng xử lý

| Thành phần | Vai trò thực tế | Trong notebook | Điều cần nhớ |
|---|---|---|---|
| **Gaussian Naive Bayes** | Mô hình phân loại, xuất dự đoán để tạo lỗi | `GaussianNB()` | Là **predictor**, không phải detector. |
| **ADWIN** | Phát hiện thay đổi thống kê trên chuỗi số bằng cửa sổ thích ứng | `ADWIN(delta=0.002)` nhận lỗi `0/1` | Có thể báo thay đổi tỷ lệ lỗi; không mặc định đồng nghĩa lỗi **tăng**. |
| **DDM** | Theo dõi mức lỗi phân loại và nhận diện suy giảm thống kê | `DDM(warm_start=30)` nhận lỗi `0/1` | Thiết kế chuyên cho error stream phân loại. |
| **Frozen / No update** | Đối chứng: model đã học ban đầu, sau đó **không học nữa** | Train 2.000 mẫu, giữ nguyên | Cho thấy hậu quả của việc không cập nhật. |
| **Drift-triggered retrain** | Thay model bằng bản được huấn luyện mới khi detector báo drift | Tạo `GaussianNB()` mới, học trên tối đa **2.000 mẫu lịch sử** | Đây là **retrain trên sliding history**, không phải `partial_fit` liên tục. |

**Một vòng xử lý trong thí nghiệm chính:**

1. Nhận `x_t`, dùng từng model dự đoán `ŷ_t`.
2. Đối chiếu nhãn thật `y_t`, ghi `correct` và tạo `error = 1 - correct`.
3. Đưa error vào detector của **chính nhánh mô hình đó**.
4. Nếu báo drift: tạo model mới, học trên tối đa 2.000 quan sát **trước** mẫu `t`, thay model cũ và reset detector.
5. Cuối cùng mới thêm mẫu `t` vào bộ đệm lịch sử. Như vậy dự đoán tại `t` không được học trước từ chính nhãn `t`.

**Chú ý:** notebook còn có một thí nghiệm **reference model online** riêng: mô hình học sau mỗi lần dự đoán; cả ADWIN và DDM cùng quan sát **một error stream tham chiếu**. Số drift ở phần này **khác** số lần retrain của hai nhánh trong thí nghiệm chính, vì mỗi nhánh có chuỗi lỗi và trạng thái detector riêng.

## 4. Kết quả đã chạy và cách đọc

### 4.1. Benchmark và giám sát tham chiếu

| Chỉ số | Kết quả | Diễn giải |
|---|---:|---|
| No-change baseline | **85,33%** (38.664/45.311) | Nhãn liên tiếp thường giữ nguyên; baseline đơn giản nhưng rất mạnh. |
| Reference prequential accuracy | **72,87%** | Accuracy của GaussianNB học tuần tự sau khi dự đoán. |
| ADWIN trên error stream tham chiếu | **74** cảnh báo | Số thời điểm ADWIN phát tín hiệu thay đổi. |
| DDM trên error stream tham chiếu | **138** cảnh báo | Số thời điểm DDM phát tín hiệu thay đổi. |

**Không suy ra DDM “tốt hơn” vì cảnh báo nhiều hơn.** Số lượng cảnh báo phản ánh độ nhạy và trạng thái detector; cần ground-truth drift hoặc thí nghiệm có điểm thay đổi biết trước mới đánh giá false alarm / detection delay.

### 4.2. Thí nghiệm chính: giữ nguyên hay retrain?

| Chiến lược | Số mẫu đánh giá | Accuracy | Số lần retrain | Chênh lệch với frozen |
|---|---:|---:|---:|---:|
| **No update (frozen)** | 43.312 | **67,01%** | 0 | — |
| **Retrain + ADWIN** | 43.312 | **73,51%** | 63 | **+6,50 điểm %** |
| **Retrain + DDM** | 43.312 | **75,81%** | 156 | **+8,80 điểm %** |

**Diễn giải đúng:** trong cấu hình đã thử, cả hai chiến lược retrain theo cảnh báo đều tăng accuracy so với model đóng băng. DDM đạt accuracy cao hơn ADWIN **2,30 điểm phần trăm**, nhưng cũng retrain nhiều hơn **93 lần**. Đây là đánh đổi **hiệu quả dự đoán – chi phí cập nhật**, chưa phải bằng chứng DDM phát hiện drift chính xác hơn.

**Không so trực tiếp** 85,33% của no-change baseline với ba nhánh trong bảng như một phép đối chứng hoàn toàn công bằng: baseline chạy trên 45.311 dự đoán, còn ba nhánh đánh giá trên 43.312 mẫu sau warm-up. Trước khi áp dụng thực tế, phải đánh giá baseline trên **cùng khoảng thời gian**; kết quả hiện tại cũng cho thấy ML chưa chắc vượt qua dự đoán nhãn trước.

### 4.3. Đọc biểu đồ

![Rolling accuracy và điểm retrain](elec2_accuracy_drift.png)

- Đường biểu diễn: accuracy của từng chiến lược trên **500 dự đoán gần nhất**, để nhìn sự tăng/giảm theo thời gian.
- Vạch/điểm đánh dấu: thời điểm **retrain thực tế** của từng nhánh khi detector báo drift.
- Không kết luận rằng mỗi lần retrain đều làm accuracy tăng ngay: đường rolling accuracy còn phụ thuộc độ khó và đặc điểm của đoạn dữ liệu kế tiếp.

![Điểm drift trên error stream tham chiếu](elec2_detector_points.png)

- Hai detector quan sát **cùng lỗi của reference model** trong biểu đồ này.
- Các dấu đánh dấu chỉ nói **detector phát cảnh báo ở vị trí đó**; Elec2 trong notebook không có nhãn “ground-truth drift point” để xác nhận cảnh báo đúng hay sai.

## 5. Chuyển từ Elec2 sang dữ liệu thật: cần thay gì?

1. **Nguồn dữ liệu và nhãn:** thay `datasets.Elec2()` bằng stream thực tế; xác định nhãn thật có sẵn ngay hay đến trễ. Không có nhãn thì **không thể tính error stream 0/1 ngay lúc dự đoán**.
2. **Thứ tự thời gian:** lưu `event_id`, `event_time`, `prediction`, `model_version` và ghép nhãn đúng sự kiện; tuyệt đối không train trước khi đánh giá mẫu tương ứng.
3. **Baseline sát nghiệp vụ:** so sánh no-change, frozen, online learning và retrain trên **cùng tập đánh giá**, không dùng độ chính xác của những khoảng dữ liệu khác nhau để xếp hạng.
4. **Tín hiệu cảnh báo:** theo dõi ADWIN/DDM và phân biệt **lỗi tăng** với thay đổi thống kê nói chung; cân nhắc cooldown, thời gian chờ nhãn và tối thiểu dữ liệu trước khi retrain.
5. **Chính sách cập nhật:** xác định kích thước lịch sử, cách xử lý nhãn trễ, cách train candidate, tiêu chí chấp thuận, versioning và rollback. Không đưa model mới lên production chỉ vì detector báo drift.
6. **Đánh giá khoa học:** đo accuracy/F1 theo thời gian, số cảnh báo, số lần retrain, chi phí tính toán và (nếu có điểm drift chuẩn) false alarms/detection delay; bổ sung periodic retraining để có benchmark mạnh hơn.

## 6. Kết luận và mức độ sẵn sàng

Task đã minh họa được luồng **label → error stream → ADWIN/DDM → retrain → đánh giá**. Trên Elec2, retrain theo cảnh báo giúp GaussianNB vượt mô hình frozen trong cùng khoảng thử nghiệm; DDM cho accuracy cao hơn nhưng đòi hỏi nhiều lần cập nhật hơn.

**Bài học lớn nhất trước dữ liệu thật:** phải kiểm soát **nhãn, thời gian, chống rò rỉ dữ liệu, baseline và chi phí cập nhật**. Drift alert là một **tín hiệu để đánh giá**, không phải lệnh bắt buộc thay model.

**Bước tiếp theo:** (1) tính no-change baseline trên cùng 43.312 mẫu; (2) so thêm periodic retraining; (3) thiết kế sơ đồ `prediction_id → label arrival → error → detector → retrain decision` cho dataset thực tế.
