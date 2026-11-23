# Báo cáo Label-based Drift Detection trên Elec2

**Đồ án:** Adaptive Model Updating in Automated Data Pipelines.
**Task:** Label Based Elec 2 
**Executor:** Võ Trọng Nghĩa

**Cơ sở báo cáo:** mã và output đang lưu trong `label_based_elec2.ipynb`, đối chiếu với JSON kết quả hiện có trong `index.html`. Lần chạy được tham chiếu: `20261123T154613+0700`, từ 2026-11-23T15:46:04.845510+07:00 đến 2026-11-23T15:46:13.273851+07:00, múi giờ Asia/Ho_Chi_Minh (UTC+07:00). Lần biên soạn này chỉ cập nhật báo cáo, không sửa code và không chạy lại thực nghiệm. “Cell” dưới đây đếm từ 1, gồm cả Markdown; tên CASE và tên bảng là tham chiếu chính.

## 1. Mục tiêu và phạm vi

Task kiểm chứng tầng giám sát lỗi có nhãn và quyết định cập nhật mô hình trong pipeline tự động. Ba câu hỏi nghiên cứu là: mô hình có vượt quy tắc dự đoán nhãn trước không; ADWIN và DDM phản ứng thế nào trên cùng chuỗi lỗi; retrain theo cảnh báo có cải thiện so với giữ nguyên và tốn bao nhiêu chi phí?

Đóng góp đã triển khai là luồng mô phỏng dự đoán → nhận nhãn → tính lỗi → giám sát → retrain có điều kiện, kèm log và phiên bản để đối soát. Thực nghiệm A cô lập detector; thực nghiệm B đánh giá chính sách cập nhật. Đây là cơ sở thiết kế tầng giám sát/cập nhật, chưa phải pipeline vận hành.

Chưa thuộc phạm vi đã hoàn thành: xử lý nhãn trễ/mất nhãn, kiểm định ứng viên trước thay mô hình, lưu trạng thái bền vững, deployment, theo dõi production và rollback. Không có mốc drift thật để xác nhận chất lượng phát hiện. Bằng chứng phạm vi: CASE 01, CASE 08–09 và phần giới hạn CASE 13.

## 2. Dữ liệu và giả định quan trọng

Elec2 được đọc qua River: **45.312 mẫu, 8 đặc trưng**, nhãn tách khỏi đặc trưng. CASE 03, cell 7 ghi không có giá trị thiếu, không có hàng đặc trưng–nhãn trùng; không áp dụng xóa trùng hay loại chuỗi giá trị lặp. Nhãn DOWN/False có 26.075 mẫu; UP/True có 19.237 mẫu, tương ứng tỷ lệ UP khoảng 42,45%.

| Nhóm trường gốc | Ý nghĩa sử dụng |
| --- | --- |
| `date`, `day`, `period` | Thông tin ngày, thứ và kỳ quan sát |
| `nswprice`, `vicprice` | Giá điện NSW và Victoria |
| `nswdemand`, `vicdemand` | Nhu cầu điện hai bang |
| `transfer` | Trao đổi điện theo lịch giữa hai bang |
| `class` → nhãn Boolean của River | UP/DOWN so với trung bình trượt giá NSW 24 giờ |

Định nghĩa nhãn và mô tả nguồn được ghi tại CASE 03, dựa trên [River Elec2](https://github.com/online-ml/river/blob/main/river/datasets/elec2.py) và [OpenML 151](https://www.openml.org/d/151). Các trường chuẩn hóa không phải số đo theo đơn vị vật lý gốc. Không diễn giải mục tiêu này thành dự báo giá kỳ tiếp theo.

Thứ tự được giữ đúng luồng River, không shuffle và không sắp xếp lại theo `date`. Tuy nhiên, `date` chuẩn hóa giảm tại mã mẫu **25.489, 34.897, 35.233, 36.241, 40.705**; chưa xác minh thời gian lịch tuyệt đối ở các điểm đó. Vì vậy, bằng chứng hiện có xác nhận thứ tự nguồn, chưa xác nhận toàn bộ timestamp lịch tăng đơn điệu. CASE 13, cell 44 có output tải ZIP nguồn trong RAM và xác nhận SHA256 CSV khớp bản đã chạy:


| Giai đoạn | Mã mẫu, tính cả hai đầu | Số mẫu | Vai trò |
| --- | --- | --- | --- |
| Huấn luyện ban đầu | 1–10.000 | 10.000 | Học pipeline cho các cấu hình ứng viên |
| Hiệu chỉnh | 10.001–20.000 | 10.000 | Chọn tốc độ học; không chọn bằng evaluation |
| Đánh giá cuối | 20.001–45.312 | 25.312 | Chấm tất cả phương án trên cùng mã mẫu |

Giả định tại thời điểm dự đoán là **mọi đặc trưng kỳ hiện tại đã có sẵn**, còn nhãn được tiết lộ **ngay sau dự đoán**. Tính sẵn có thực tế của giá và nhu cầu cùng kỳ chưa được nguồn xác nhận. Đây là điều kiện benchmark cần xác minh lại trước ứng dụng thật.

StandardScaler bổ sung chỉ học dữ liệu được phép: trước evaluation khi khởi tạo, sau dự đoán đối với học từng mẫu, hoặc từ buffer đã có nhãn khi retrain. Chưa có dữ liệu thô để xác minh thống kê dùng trong chuẩn hóa gốc của Elec2; không thể kết luận toàn bộ tiền xử lý upstream đã loại trừ rò rỉ thời gian. Biểu đồ **CASE 03 — Phân bố nhãn** và **CASE 03 — Tỷ lệ UP theo đoạn** chỉ mô tả dữ liệu, không tự xác nhận concept drift.

## 3. Quy trình đã triển khai

Sau kiểm tra dữ liệu, notebook chia ba đoạn liên tiếp và thử đúng ba tốc độ học 0,01; 0,05; 0,10. Accuracy hiệu chỉnh lần lượt là 0,5045; 0,4996; 0,4997; chọn **0,01**. Sau khóa cấu hình, pipeline mới được học trên toàn bộ 20.000 mẫu trước evaluation, rồi `deepcopy` toàn bộ scaler và classifier cho các nhánh. Detector khởi tạo rỗng; metric đánh giá bắt đầu lại từ đầu evaluation. Bằng chứng: CASE 04, cell 13 và **CASE 04 — Phân chia thời gian**.

| Phương án | Dự đoán | Chính sách học trong evaluation |
| --- | --- | --- |
| Nhãn trước | Nhãn thật của mẫu liền trước; khởi tạo bằng nhãn mẫu 20.000 | Chỉ ghi nhớ nhãn mới sau khi chấm, không có classifier |
| Giữ nguyên | Pipeline đã học trước evaluation | Không cập nhật cả scaler lẫn classifier |
| Học từng mẫu | Pipeline hiện tại | Dự đoán → chấm → `learn_one` mỗi mẫu |
| ADWIN | Pipeline riêng của nhánh | Chỉ retrain khi ADWIN báo drift và đủ điều kiện |
| DDM | Pipeline riêng của nhánh | Chỉ retrain khi DDM báo drift và đủ điều kiện |

Baseline đo lợi thế từ tính liên tục của nhãn, khác mô hình giữ nguyên. Test-then-train là thứ tự xử lý; chính sách quyết định khi nào học. Nhánh học từng mẫu thể hiện prequential đầy đủ và được đánh giá riêng với các nhánh chỉ retrain.

**Cấu hình chính, CASE 02 và CASE 04:** seed 42; StandardScaler → LogisticRegression, optimizer SGD với tốc độ học 0,01, `l2=0.001`; ADWIN `delta=0.002` và mặc định API `clock=32`, `max_buckets=5`, `min_window_length=5`, `grace_period=10`; DDM `warm_start=100`, `warning_threshold=2.0`, `drift_threshold=3.0`. Các ngưỡng detector được định trước, không hiệu chỉnh trên evaluation.

Cửa sổ metric là **500 mẫu đã chấm**, đầu evaluation có thể ngắn hơn; buffer retrain là **2.000 mẫu có nhãn gần nhất**. Cần ít nhất **1.000 mẫu** và cách đầu evaluation hoặc lần retrain trước ít nhất **1.000 mẫu**. Buffer ban đầu lấy từ dữ liệu trước evaluation. Sau dự đoán/chấm mẫu hiện tại, nhánh thêm mẫu vào buffer và cập nhật detector bằng lỗi của chính mình. Nếu đủ điều kiện, tạo mới cả scaler và classifier, học trên buffer, tăng phiên bản và thay detector bằng đối tượng rỗng; mô hình mới chỉ dự đoán từ mẫu kế tiếp. Nếu thiếu điều kiện, ghi hoãn và giữ phiên bản. Warning không kích hoạt học.

Trong A, để detector tự reset tại update kế sau drift theo API River. Trong B, sau retrain thành công thay detector rỗng ngay; nếu hoãn thì để API tự reset ở update kế. Không có lần reset thủ công bổ sung trên detector mới. Quy tắc ADWIN hiện tại áp dụng cho mọi cảnh báo, kể cả chiều lỗi giảm; đây là chính sách đã thử, chưa có bộ lọc riêng cho suy giảm.

| Bước xử lý | Đã thực hiện | Bằng chứng | Trạng thái |
| --- | --- | --- | --- |
| Kiểm tra môi trường/API | Ghi phiên bản, cấu hình và hành vi reset | CASE 02, cell 4 | Đã chạy |
| Load và kiểm tra dữ liệu | Schema, nhãn, thiếu, mã mẫu, thứ tự, hash nguồn | CASE 03, cell 7; CASE 13, cell 44 | Đã chạy và đối soát |
| Chia và hiệu chỉnh | Ba đoạn liên tiếp, ba ứng viên, khóa cấu hình | CASE 04, cell 12–13 | Đã chạy và kiểm tra khóa |
| Khởi tạo các phương án | Baseline và các bản sao pipeline đã học | CASE 05–06, cell 15,17; CASE 09, cell 24 | Đã chạy, kiểm tra độc lập |
| So sánh detector | Cùng reference error stream | CASE 07, cell 19–21; CASE 12 | Đã chạy, hash chuỗi khớp |
| So sánh cập nhật | Lỗi riêng từng nhánh, buffer, hoãn, phiên bản | CASE 09, cell 24 | Đã chạy và kiểm chứng logic |
| Metric và phân tích phiên bản | Tổng hợp, cửa sổ, tích lũy, trước/sau | CASE 10–11, cell 28–38 | Có output và biểu đồ |
| Kiểm tra bất biến | 163 kiểm tra đạt; dữ liệu thử tách benchmark | CASE 12, cell 40 | Đã chạy và đạt trong mô phỏng |
| Đồng bộ trình bày | Báo cáo/JSON HTML và kiểm thử mã hiển thị | CASE 13, cell 44 | Đã chạy; chưa kiểm thử trình duyệt thật |

Môi trường của output đang dùng: Python 3.11.11, River 0.23.0, NumPy 2.4.6, pandas 2.3.3, Matplotlib 3.11.2, nbformat 5.11.1, nbclient 0.11.0 và ipykernel 7.4.0; DDM import từ `river.drift.binary` (CASE 02). Đây là phiên bản của lần chạy đã lưu, không phải suy đoán từ môi trường lúc viết báo cáo.

## 4. Thực nghiệm A — So sánh ADWIN và DDM

Reference error stream lấy duy nhất từ mô hình **giữ nguyên**, mã mẫu 20.001–45.312; 0 là đúng, 1 là sai. Mô hình tham chiếu không thay sau cảnh báo. Hai detector đọc cùng bảng lỗi theo cùng thứ tự; CASE 12 kiểm tra SHA256 hai chuỗi bằng nhau và bằng chuỗi lỗi tham chiếu. Warning được ghi khi bắt đầu đợt, không phải mỗi mẫu đang ở trạng thái warning.

Kết quả **CASE 07, bảng `bang_su_kien_a`**, output cell 19 và nhận xét cell 21:

| Detector | Loại sự kiện | Số sự kiện | Lỗi cửa sổ trung bình |
| --- | --- | --- | --- |
| ADWIN | Drift | 57 | 0.353151 |
| DDM | Drift | 65 | 0.360702 |
| DDM | Warning | 185 | 0.370232 |

ADWIN không có API warning nên trường này **không áp dụng**, không diễn giải là detector đã theo dõi warning nhưng ghi nhận 0 lần. Mức lỗi trong bảng là trung bình tỷ lệ lỗi cửa sổ tại các sự kiện tương ứng, không phải precision detector.

Đối chiếu **CASE 07 — Lỗi tham chiếu và sự kiện detector** cho thấy cảnh báo không chỉ xuất hiện khi lỗi tăng. Ví dụ ADWIN tại mẫu 20.256 có lỗi cửa sổ 0,515625, ước lượng trước/sau update là 0,513725/0,562500; tại mẫu 20.608, lỗi cửa sổ 0,390000 nhưng ước lượng giảm từ 0,452632 xuống 0,337500. Toàn A có **36 cảnh báo chiều tăng, 21 chiều giảm**, phân loại theo chênh ước lượng trước/sau update và cắt cửa sổ; đó là chẩn đoán chiều, không phải kiểm định xu hướng độc lập.

Có **56/57** cảnh báo ADWIN nằm trong phạm vi ±500 mẫu của ít nhất một drift DDM (cell 21). Điều này mô tả sự gần nhau về vị trí, chưa chứng minh hai detector nhận diện cùng một drift thật hoặc cho phép gọi khoảng cách đó là độ trễ phát hiện.

Có thể kết luận hai cơ chế đưa ra lịch cảnh báo khác nhau trên cùng lỗi; ADWIN phản ứng hai chiều. Chưa đủ bằng chứng xếp hạng detector: không có ground truth drift, chưa đo precision/recall hoặc độ trễ đã xác nhận. Nhiều cảnh báo hơn không có nghĩa tốt hơn; không báo cũng không chứng minh không có drift.

## 5. Thực nghiệm B — So sánh chính sách cập nhật

Các nhánh được chấm trên cùng 25.312 mã mẫu, cùng quy tắc metric. Các nhánh ML dùng cùng thuật toán và trạng thái khởi tạo đã học; hai nhánh adaptive có mô hình/detector độc lập, cùng buffer và điều kiện retrain. Mỗi detector B nhận lỗi của chính nhánh đó, nên số cảnh báo B không dùng để cô lập chất lượng detector. Baseline là quy tắc đối chứng riêng, không có pipeline khởi tạo.

Bảng **CASE 11 — `bang_tong_hop`**, cell 38. Accuracy và balanced accuracy ở thang 0–1; balanced accuracy là trung bình recall hai nhãn. Các chênh lệch là **điểm phần trăm**, tính trước khi làm tròn. Thời gian là tổng giây tạo và huấn luyện lại pipeline bằng `perf_counter`.

| Phương án | Số mẫu được chấm | Accuracy | Balanced accuracy | Chênh với baseline (đpt) | Chênh với giữ nguyên (đpt) | Số drift | Số retrain | Tổng thời gian retrain (giây) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nhãn trước | 25312 | 0.857814 | 0.853535 | 0.000000 | 20.642383 | Không áp dụng | Không áp dụng | Không áp dụng |
| Giữ nguyên | 25312 | 0.651391 | 0.649194 | -20.642383 | 0.000000 | Không áp dụng | Không áp dụng | Không áp dụng |
| Học từng mẫu | 25312 | 0.818505 | 0.805445 | -3.930942 | 16.711441 | Không áp dụng | Không áp dụng | Không áp dụng |
| ADWIN | 25312 | 0.688488 | 0.679925 | -16.932680 | 3.709703 | 45 | 17 | 0.766926 |
| DDM | 25312 | 0.695441 | 0.673471 | -16.237358 | 4.405025 | 62 | 17 | 0.760072 |

Phân biệt cảnh báo với hành động, từ cùng bảng và **CASE 10 — Biểu đồ 3: Timeline drift và retrain**:

| Phương án | Số warning | Số drift | Số retrain | Số hoãn retrain |
| --- | --- | --- | --- | --- |
| ADWIN | Không áp dụng | 45 | 17 | 28 |
| DDM | 203 | 62 | 17 | 45 |

Hai nhánh kết thúc ở phiên bản 18: bản khởi tạo cộng 17 retrain. Với ADWIN, 45 drift = 17 retrain + 28 hoãn; DDM, 62 = 17 + 45. Warning DDM là 203 lần bắt đầu đợt; không cộng warning vào số lần cập nhật. “Không áp dụng” khác giá trị 0 và khác thiếu kết quả.

**Chất lượng theo thời gian.** Trong **CASE 10 — Biểu đồ 1: Accuracy cửa sổ**, cửa sổ cuối đạt baseline 0,8300, giữ nguyên 0,4620, học từng mẫu 0,9040, ADWIN 0,8020 và DDM 0,8620. Vì vậy, DDM/học từng mẫu có thể vượt baseline ở đoạn cuối dù thua trên toàn evaluation. **CASE 10 — Biểu đồ 2: Accuracy tích lũy** cho tổng thể, nhưng có thể che suy giảm gần đây.

**CASE 10 — Biểu đồ 4: Chênh lệch accuracy cửa sổ** ghi ADWIN cao hơn giữ nguyên tại 50,55% vị trí cửa sổ, DDM tại 45,42%; chênh cuối đoạn lần lượt +34,00 và +40,00 điểm phần trăm. Đây là các cửa sổ chồng lấn, không phải tỷ lệ thử nghiệm độc lập hay xác suất cập nhật thành công.

**Lợi ích và chi phí.** Trên toàn evaluation, adaptive cao hơn giữ nguyên +3,709703 và +4,405025 điểm phần trăm nhưng vẫn thấp hơn baseline lần lượt 16,932680 và 16,237358 điểm phần trăm; cũng thấp hơn nhánh học từng mẫu. Mỗi nhánh retrain 17 lần với tổng thời gian khoảng 0,77 giây trong lần chạy này. **CASE 10 — Biểu đồ 5: Chất lượng và chi phí** chỉ đo retrain, không tính thời gian detector, dự đoán, học từng mẫu, khởi tạo hoặc vẽ; chưa đủ căn cứ so tổng chi phí vận hành hay kết luận DDM nhanh hơn một cách ổn định.

**Sau từng cập nhật.** CASE 11 dùng tối đa 500 mẫu trước/sau, cắt đoạn sau tại retrain kế và đoạn trước tại retrain trước để không trộn phiên bản. Toàn bộ 34 lần trong output hiện có đủ 500 mẫu mỗi phía. ADWIN có 7 đoạn sau thắng giữ nguyên, 9 đoạn thua, 1 ngang; DDM có 8 thắng, 9 thua. Các ví dụ chọn theo chênh cao nhất/thấp nhất mỗi detector và thêm mẫu 36.672 để kiểm tra cách diễn giải trước/sau:

| Mã mẫu | Detector | Phiên bản mới | Số mẫu trước | Số mẫu sau | Accuracy trước | Accuracy sau | Accuracy giữ nguyên cùng đoạn sau | Chênh với giữ nguyên (đpt) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 34272 | ADWIN | 12 | 500 | 500 | 0.578000 | 0.444000 | 0.704000 | -26.000000 |
| 36672 | ADWIN | 13 | 500 | 500 | 0.386000 | 0.804000 | 0.808000 | -0.400000 |
| 44032 | ADWIN | 18 | 500 | 500 | 0.796000 | 0.850000 | 0.438000 | 41.200000 |
| 25192 | DDM | 5 | 500 | 500 | 0.684000 | 0.610000 | 0.802000 | -19.200000 |
| 41794 | DDM | 16 | 500 | 500 | 0.818000 | 0.876000 | 0.508000 | 36.800000 |

Cập nhật có ích rõ về **so sánh quan sát trên cùng đoạn** ở ADWIN mẫu 44.032 (+41,2 điểm phần trăm), DDM mẫu 41.794 (+36,8); chưa có ích ở ADWIN mẫu 34.272 (−26,0), DDM mẫu 25.192 (−19,2). Tại ADWIN mẫu 36.672, accuracy tăng từ 0,386 lên 0,804 sau retrain, nhưng giữ nguyên đạt 0,808 trên chính đoạn sau: tăng trước/sau vẫn chưa chứng minh cập nhật tốt hơn giữ nguyên. Các ví dụ cực trị chỉ minh họa độ biến thiên, không đại diện lợi ích trung bình hay quan hệ nhân quả.

## 6. Những điểm đã hoàn thành

Notebook lưu **20/20 cell code đã thực thi**, không có output lỗi; CASE 12, cell 40 hiển thị **163 kiểm tra đạt**. Đây là bằng chứng của lần chạy lưu sẵn; nhiệm vụ viết báo cáo không kiểm thử lại môi trường hay thực nghiệm.

| Nội dung xác nhận | Mức bằng chứng | Vị trí trong notebook |
| --- | --- | --- |
| Chạy đủ baseline, giữ nguyên, học từng mẫu và hai adaptive | Có code, đã chạy; đủ cùng mã mẫu được kiểm tra | CASE 05–06,09 và CASE 12 |
| Giữ nguyên cả mô hình/preprocessing; bản sao học không ảnh hưởng nhánh khác | Hash trạng thái và kiểm thử cập nhật riêng đạt | CASE 06; CASE 12, cell 40 |
| Không chọn cấu hình trên evaluation | Mã hiệu chỉnh chỉ dùng đoạn giữa; cấu hình khóa không đổi | CASE 04 và CASE 12 |
| Cùng lỗi trong A; lỗi riêng trong B | Hash chuỗi A khớp, đường xử lý B riêng | CASE 07,09,12 |
| Accuracy, balanced accuracy và số sự kiện đúng log | Tính lại từ dự đoán, đối soát từng loại sự kiện và tổng thời gian | CASE 11–12 |
| Retrain chỉ dùng phạm vi đã có nhãn, đủ điều kiện, có hiệu lực từ mẫu kế | Assert phạm vi/phiên bản; kiểm thử logic nhiều drift và hoãn | CASE 09,12 |
| Đảo nhãn hiện tại không đổi dự đoán hiện tại | Kiểm thử trên hàm adaptive với mỗi detector đạt | CASE 12; không suy rộng thành kiểm chứng mọi trạng thái production |
| Không drift, nhiều drift, thiếu mẫu và dữ liệu rỗng | Kiểm thử nhỏ trong bộ nhớ tách riêng kết quả benchmark | CASE 12 |
| Trình bày kết quả website | Cell đồng bộ đã chạy; Node kiểm tra 6 tab, 5 dòng kết quả, CSV và ghi chú riêng | CASE 13, cell 44; chỉ DOM giả lập |

Những sửa đổi có bằng chứng trong bảng rà soát CASE 13: khai báo schema cố định cho log rỗng; dùng chung hàm adaptive cho benchmark/kiểm thử; bổ sung kiểm tra độc lập, đảo nhãn và điều kiện hoãn; thêm thống kê warning/hoãn; phân biệt warning ADWIN không hỗ trợ bằng “Không áp dụng”/`null`; thay số nhập tay website bằng một JSON từ kết quả đã tính. Không có bằng chứng những sửa đổi này thay ngưỡng để tăng accuracy.

Bản báo cáo này đồng bộ lại thời gian retrain và số kiểm tra theo output hiện hành thay cho số liệu cũ. Không sửa notebook hoặc website. SHA256 bảng dự đoán ghi tại CASE 12: `469d2395869022669dc13f83a696289a04e410591e8591331477574cd2f6b14a`.

## 7. Vấn đề còn tồn tại và lưu ý cần nhắc lại

Các mục dưới đây phân biệt giới hạn/giả định với lỗi đã được chứng minh; chưa có cơ sở coi tất cả là lỗi thực nghiệm.

| Vấn đề | Bằng chứng | Ảnh hưởng | Cách xử lý đề xuất | Mức ưu tiên |
| --- | --- | --- | --- | --- |
| Đặc trưng cùng kỳ và mục tiêu nhãn | CASE 03 giả định giá/nhu cầu đã có; mục tiêu UP/DOWN theo trung bình 24 giờ | Có thể không phù hợp thời điểm dự đoán production | Xác nhận thời điểm công bố và mục tiêu nghiệp vụ; thiết kế lại đặc trưng/mục tiêu nếu cần | Cao, trước tích hợp |
| Nhãn trễ/mất nhãn | CASE 01,03 dùng nhãn ngay sau dự đoán | Cảnh báo, buffer và độ mới mô hình sẽ khác khi nhãn đến muộn | Hàng đợi dự đoán, ghép nhãn theo mã; chỉ chấm/học khi nhãn tới | Cao |
| Chuẩn hóa gốc và date bất thường | CASE 03 có 5 vị trí date giảm; không có dữ liệu thô/thống kê upstream | Chưa xác minh đầy đủ nhân quả tiền xử lý và lịch sự kiện | Thu thập dữ liệu thô, timestamp; scaler chỉ fit dữ liệu trước thời điểm dùng | Cao |
| Phụ thuộc thời gian | Baseline nhãn trước đạt 85,78%; các metric cửa sổ chồng lấn | Không coi mẫu/cửa sổ là thử nghiệm độc lập | Đánh giá trên nhiều đoạn liên tiếp, dùng cách lượng hóa bất định phù hợp phụ thuộc thời gian | Vừa |
| Drift khác retrain | CASE 11: 45/62 drift nhưng mỗi nhánh chỉ 17 retrain | Đếm cảnh báo như cập nhật sẽ sai chi phí/phiên bản | Giữ riêng warning, drift, retrain, hoãn và lý do | Cao; logic mô phỏng đã có |
| Buffer, khoảng chờ và reset | CASE 08–09: 2.000/1.000/1.000 mẫu; detector reset sau cập nhật hoặc theo API khi hoãn | Ảnh hưởng độ mới dữ liệu và tần suất phản ứng; ADWIN giảm lỗi vẫn có thể kích hoạt | Giữ cấu hình làm mốc, kiểm thử nhãn trễ; thay chính sách chỉ qua hiệu chỉnh/đánh giá mới | Vừa |
| Chất lượng mô hình sau cập nhật | CASE 11 có 9 đoạn thua giữ nguyên ở mỗi detector; toàn evaluation thua baseline | Tự thay mô hình chưa có bảo đảm lợi ích | Thêm cổng kiểm định ứng viên và đối chứng cùng đoạn trước thay thế | Cao |
| Thiếu mốc drift thật | CASE 07,12 không có ground truth | Chưa xếp hạng chất lượng phát hiện hay đo độ trễ xác nhận | Bổ sung nguồn nhãn sự kiện đáng tin cậy hoặc benchmark riêng có drift biết trước | Vừa |
| Chi phí chưa toàn diện | CASE 10 — Biểu đồ 5 chỉ đo retrain | Chưa định lượng độ trễ/CPU/RAM toàn pipeline | Đo riêng dự đoán, detector, huấn luyện, bộ nhớ và p95/p99 khi tích hợp | Vừa |
| Trình bày và lưu báo cáo | CASE 13 kiểm thử DOM giả lập; cell cuối sinh lại báo cáo từ mẫu trong code | Chưa xác minh giao diện thật; Run All tương lai có thể ghi đè cấu trúc báo cáo biên soạn này | Kiểm thử trình duyệt và thống nhất cơ chế bảo toàn tài liệu trước lần chạy kế; nhiệm vụ này không sửa code | Vừa |

## 8. Bàn giao cho giai đoạn tiếp theo

Cần kế thừa schema tám trường và nhãn Boolean, quy ước mã mẫu, cấu hình khóa, trạng thái pipeline đã học, trạng thái detector và buffer. Mã mẫu hiện tại là vị trí luồng; **thời gian sự kiện thực, thời gian dự đoán và thời gian nhận nhãn chưa được mô hình hóa thành bộ trường riêng**. Không thay các trường đó bằng `date` chuẩn hóa khi tích hợp.

Đầu ra hiện có trong bộ nhớ gồm `bang_du_doan` (mã mẫu, phương án, nhãn thật, dự đoán, lỗi, phiên bản, accuracy tích lũy/cửa sổ) và `bang_su_kien` (mã mẫu, phương án/detector, loại sự kiện, phiên bản cũ/mới, đầu/cuối và số mẫu retrain, thời gian, lý do hoãn); A có `bang_su_kien_a` riêng. Phải giữ dự đoán gắn với phiên bản **đã thực sự dự đoán**, không chấm lại lịch sử bằng phiên bản mới.

JSON trong HTML lưu cấu hình, phiên bản thư viện, kiểm tra, sự kiện và chuỗi metric; không thay thế kho trạng thái mô hình hay log dự đoán bền vững. Notebook chưa bàn giao snapshot model/scaler/detector để phục hồi sau sự cố. Khi tiếp tục, cần giữ cả schema, cấu hình, scaler, classifier, detector, buffer và metric cùng metadata phiên bản; không chỉ giữ trọng số classifier.

| Hạng mục tiếp theo | Đầu vào kế thừa | Việc cần làm | Tiêu chí nghiệm thu | Ưu tiên |
| --- | --- | --- | --- | --- |
| Hợp đồng dữ liệu và thời gian | Schema CASE 03, mã mẫu và giả định | Bổ sung thời gian sự kiện/dự đoán/nhận nhãn, nguồn công bố từng trường | Mọi mẫu có khóa ghép duy nhất; không dùng trường xuất hiện sau dự đoán | Cao |
| Ghép nhãn trễ | Log dự đoán, baseline và thứ tự chấm | Hàng đợi, nhãn đến sai thứ tự, trùng, mất và sửa nhãn | Không dự đoán lại quá khứ; mỗi nhãn chỉ chấm/cập nhật đúng số lần quy định | Cao |
| Lưu trạng thái và phục hồi | Pipeline, detector, buffer, metric, cấu hình | Snapshot theo phiên bản, khôi phục đồng bộ | Sau khôi phục từ cùng trạng thái, phát lại cùng luồng cho cùng dự đoán/sự kiện | Cao |
| Điều phối cập nhật | Quy tắc drift/hoãn CASE 08–09 | Giữ phân biệt các sự kiện; bổ sung điều phối chống retrain trùng | Drift không đủ điều kiện không tạo phiên bản mới; mô hình mới chỉ có hiệu lực sau điểm chuyển rõ ràng | Cao |
| Kiểm định ứng viên | Pipeline hiện hành và mô hình mới từ buffer | Thiết kế đoạn kiểm định hợp lệ theo thời gian, đối chứng baseline/giữ nguyên, ngưỡng chấp nhận và giới hạn chi phí | Chỉ triển khai ứng viên qua tiêu chí định trước; không chọn bằng nhãn tương lai hoặc tinh chỉnh lại evaluation cũ | Cao |
| Theo dõi sau triển khai và rollback | Metric cửa sổ/tích lũy, log sự kiện, phiên bản | Theo dõi chất lượng khi nhãn tới, tỷ lệ thiếu nhãn, độ trễ; giữ phiên bản ổn định để phục hồi | Diễn tập phát hiện suy giảm và khôi phục đúng model cùng scaler; log đầy đủ quyết định | Cao |
| Đánh giá mở rộng | Bộ cấu hình và kết quả mốc hiện tại | Thử thêm đoạn thời gian, nhãn trễ, chi phí toàn tuyến; mốc drift biết trước ở bài thử riêng | Tách hiệu chỉnh/đánh giá, không trộn dữ liệu mô phỏng với Elec2 | Vừa |
| Hoàn thiện trang bàn giao | JSON HTML và kiểm thử Node đã có | Kiểm thử file HTML trên trình duyệt rộng/hẹp, CSV thực tế và dữ liệu lưu cũ | Đồ thị, tab, checklist và CSV khớp đúng lần chạy; trạng thái chưa xác minh hiển thị rõ | Vừa |

Các hàng trên là **việc cần phát triển**, không phải chức năng deployment/rollback đã hoàn thành. Bằng chứng hiện có mới đến mô phỏng retrain và đối soát; CASE 13 cũng ghi chưa kiểm thử trình duyệt thật, chưa xuất bản website.

## 9. Kết luận

1. **Mô hình có vượt baseline không?** Chưa. Baseline đạt 85,78%; học từng mẫu đạt 81,85%, ADWIN 68,85%, DDM 69,54% và giữ nguyên 65,14% trên cùng evaluation.
2. **Detector phản ứng thế nào?** Trên cùng lỗi giữ nguyên, ADWIN báo 57 drift với cả chiều tăng/giảm; DDM báo 65 drift và 185 đợt warning. Đây là khác biệt hành vi đã đo, chưa đủ bằng chứng detector nào phát hiện drift thật tốt hơn.
3. **Cập nhật có hiệu quả không?** Trong cấu hình này, ADWIN/DDM tăng accuracy tổng so giữ nguyên lần lượt 3,71/4,41 điểm phần trăm với 17 retrain mỗi nhánh. Lợi ích không đồng đều theo thời gian, có các đoạn sau cập nhật thua đối chứng và chưa vượt baseline tổng thể; chưa thể cam kết hiệu quả production.

Task đủ làm mốc thực nghiệm và đầu vào thiết kế tầng giám sát/cập nhật tiếp theo. Chưa sẵn sàng tự động thay mô hình trong pipeline thực tế trước khi xác minh dữ liệu–thời gian, xử lý nhãn trễ, lưu/khôi phục trạng thái và thiết lập cổng kiểm định ứng viên cùng rollback.

**Thông tin còn thiếu cần bổ sung:**

- Thời điểm công bố đặc trưng, timestamp lịch gốc và quy trình chuẩn hóa upstream.
- Phân bố độ trễ, mất và sửa nhãn trong nguồn dữ liệu thực tế.
- Mốc drift thật hoặc nguồn xác nhận sự kiện đáng tin cậy.
- Ngưỡng chấp nhận ứng viên, giới hạn chi phí/độ trễ và điều kiện rollback.
- Hợp đồng lưu trạng thái, môi trường triển khai và bằng chứng kiểm thử trình duyệt thật.
