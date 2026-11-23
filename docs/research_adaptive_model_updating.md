# Nghiên cứu phương pháp cập nhật thích ứng mô hình học máy trong quy trình dữ liệu tự động

**Người thực hiện:** Nguyễn Tố Quyên (`Quyen0005`)  
**Dự án:** Adaptive Model Updating in Automated Data Pipelines  

---

## 1. Giới thiệu vấn đề nghiên cứu

Trong các hệ thống học máy được triển khai thực tế, dữ liệu thường xuyên thay đổi theo thời gian. Sự thay đổi này có thể khiến mô hình được huấn luyện trên dữ liệu cũ dần mất khả năng dự đoán chính xác. Hiện tượng này thường được gọi là **Concept Drift**.

Để duy trì hiệu suất của mô hình, hệ thống cần có khả năng theo dõi sự thay đổi của dữ liệu, phát hiện dấu hiệu suy giảm hiệu suất và quyết định thời điểm cập nhật hoặc huấn luyện lại mô hình (**Model Retraining**). Tuy nhiên, việc huấn luyện lại quá thường xuyên có thể làm tăng chi phí tính toán, thời gian xử lý và tài nguyên hệ thống. Ngược lại, nếu cập nhật quá chậm, mô hình có thể tiếp tục đưa ra dự đoán kém chính xác.

Vì vậy, vấn đề nghiên cứu không chỉ nằm ở việc phát hiện Concept Drift mà còn ở việc xác định:
- Khi nào cần cập nhật mô hình?
- Cập nhật bằng phương pháp nào?
- Làm thế nào để cân bằng giữa hiệu suất dự đoán với chi phí vận hành?

Đề tài tập trung nghiên cứu các phương pháp cập nhật thích ứng mô hình học máy trong quy trình dữ liệu tự động, hướng đến khả năng duy trì hiệu suất mô hình khi dữ liệu thay đổi theo thời gian, đồng thời cân nhắc các giới hạn về tài nguyên và độ trễ của hệ thống.

---

## 2. Cơ sở lý thuyết

| Khái niệm | Nội dung | Vai trò trong nghiên cứu |
| :--- | :--- | :--- |
| **Concept Drift** | Sự thay đổi của mối quan hệ hoặc phân phối dữ liệu theo thời gian, có thể khiến mô hình hiện tại suy giảm hiệu suất. | Xác định vấn đề cần phát hiện và xử lý. |
| **Data Drift** | Sự thay đổi trong phân phối dữ liệu đầu vào theo thời gian. | Theo dõi sự thay đổi của dữ liệu trong pipeline. |
| **Drift Detection** | Quá trình sử dụng các phương pháp thống kê hoặc chỉ số hiệu suất để phát hiện sự thay đổi. | Cung cấp tín hiệu cho quá trình đánh giá và cập nhật mô hình. |
| **Model Retraining** | Huấn luyện lại mô hình bằng dữ liệu mới hoặc dữ liệu gần đây. | Khôi phục hoặc cải thiện hiệu suất dự đoán. |
| **Incremental Learning** | Cập nhật mô hình từng phần khi có dữ liệu mới, thay vì huấn luyện lại toàn bộ. | Là một hướng thích ứng có thể giảm nhu cầu retraining toàn phần. |
| **Adaptive Learning** | Khả năng điều chỉnh mô hình hoặc chiến lược học tập để thích nghi với môi trường dữ liệu thay đổi. | Là định hướng tổng thể của đề tài. |
| **Prequential Evaluation** | Đánh giá theo trình tự: dự đoán trước, sau đó mới sử dụng nhãn thực tế để tính lỗi và cập nhật. | Phù hợp với việc đánh giá mô hình trên luồng dữ liệu tuần tự. |
| **Retraining Cost** | Chi phí tài nguyên, thời gian và xử lý phát sinh khi huấn luyện lại mô hình. | Là yếu tố cần cân nhắc khi lựa chọn chính sách cập nhật. |

---

## 3. Phân tích hai nghiên cứu chính

### 3.1. Nghiên cứu 1: When to Retrain: An Empirical Study of Retraining Policies for Streaming ML Under Concept Drift, Budget, and Latency Constraints (2026)

| Tiêu chí | Nội dung phân tích |
| :--- | :--- |
| **Tác giả** | Sawan Dasari |
| **Mục tiêu** | Nghiên cứu thời điểm phù hợp để huấn luyện lại mô hình trong môi trường dữ liệu luồng có Concept Drift, ngân sách retraining giới hạn và độ trễ huấn luyện, triển khai. |
| **Phương pháp** | So sánh Periodic Retraining, Error-threshold Retraining, ADWIN-triggered Retraining và No-retrain. |
| **Dữ liệu thực nghiệm** | Dữ liệu tổng hợp, LUFlow và LendingClub. |
| **Thiết lập thực nghiệm** | 3 dạng drift: abrupt, gradual và recurring; nhiều mức ngân sách và độ trễ; so sánh mô hình có và không có Incremental Learning. |
| **Kết quả chính** | Khi mô hình có cập nhật từng mẫu bằng Incremental Learning, sự khác biệt thực tiễn giữa các chính sách retraining không đáng kể trong phạm vi mô hình tuyến tính và nhãn đến ngay được nghiên cứu. Khi không có Incremental Learning, lựa chọn chính sách retraining ảnh hưởng đáng kể đến hiệu suất sau drift. |
| **Phát hiện đáng chú ý** | Periodic Retraining hoạt động tốt hơn các phương pháp phản ứng trong nhiều trường hợp abrupt và gradual drift ở thiết lập không có Incremental Learning. Các phương pháp phản ứng vẫn có lợi thế trong một số trường hợp recurring drift. |
| **Hạn chế** | Kết quả phụ thuộc vào mô hình tuyến tính, cơ chế cập nhật và giả định nhãn đến ngay. Không thể mặc định kết quả sẽ giống nhau với mô hình khác hoặc môi trường có nhãn đến trễ. |
| **Giá trị đối với đề tài** | Cung cấp cơ sở để so sánh các chính sách cập nhật mô hình, đồng thời cho thấy cần đánh giá cả Incremental Learning, chi phí retraining và độ trễ triển khai. |

> **Nhận xét:** Nghiên cứu cho thấy không có một chính sách retraining tối ưu cho mọi tình huống. Hiệu quả phụ thuộc vào loại drift, khả năng cập nhật từng phần, ngân sách và độ trễ. Vì vậy, đề tài cần đánh giá chính sách cập nhật trong điều kiện vận hành cụ thể thay vì chỉ dựa vào độ chính xác của mô hình.

### 3.2. Nghiên cứu 2: Cost-Effective Retraining of Machine Learning Models / Cost-aware retraining for machine learning (2023–2024)

| Tiêu chí | Nội dung phân tích |
| :--- | :--- |
| **Tác giả** | Ananth Mahadevan và Michael Mathioudakis |
| **Công bố** | Bản tiền xuất bản năm 2023; bài báo đăng trên Knowledge-Based Systems năm 2024 với tên Cost-aware retraining for machine learning. |
| **Mục tiêu** | Xây dựng phương pháp quyết định có nên huấn luyện lại mô hình hay tiếp tục sử dụng mô hình hiện tại dựa trên sự cân bằng giữa chi phí mô hình lỗi thời và chi phí retraining. |
| **Phương pháp** | Đề xuất Cost-Aware Retraining Algorithm (CARA), với các biến thể CARA-T, CARA-CT và CARA-P. |
| **Cơ sở quyết định** | Xem xét mức suy giảm hiệu suất trên các truy vấn dự đoán và chi phí tài nguyên cần thiết để huấn luyện lại. |
| **Dữ liệu thực nghiệm** | Các bộ dữ liệu tổng hợp và dữ liệu thực tế như Electricity, Covertype và Airlines. |
| **Kết quả chính** | CARA có khả năng cân bằng chi phí và hiệu suất; trong các thực nghiệm được báo cáo, CARA đạt độ chính xác truy vấn tương đương hoặc tốt hơn các phương pháp phát hiện drift cơ sở trong khi đưa ra ít quyết định retraining hơn ở nhiều trường hợp. |
| **Phát hiện đáng chú ý** | CARA-P là chính sách cập nhật định kỳ, có thể hoạt động tốt khi dữ liệu hoặc truy vấn có tính chu kỳ nhưng có thể gây retraining không cần thiết khi không tồn tại tính chu kỳ rõ ràng. |
| **Hạn chế** | Việc tính toán độ tương đồng theo cặp có thể tốn tài nguyên khi dữ liệu lớn; chi phí retraining trong nghiên cứu được giả định cố định ở một số thiết lập. |
| **Giá trị đối với đề tài** | Cung cấp cơ sở để nghiên cứu chính sách cập nhật không chỉ dựa trên tín hiệu drift mà còn cân nhắc chi phí và lợi ích thực tế của việc retraining. |

> **Nhận xét:** Nghiên cứu mở rộng cách tiếp cận từ phát hiện drift sang ra quyết định cập nhật có xét đến chi phí. Đây là cơ sở quan trọng để đề tài xem xét thêm tài nguyên, thời gian huấn luyện và mức cải thiện hiệu suất khi lựa chọn cập nhật mô hình.

---

## 4. Tổng hợp năm nghiên cứu bổ trợ

| STT | Nghiên cứu | Nội dung chính | Giá trị đối với đề tài |
| :---: | :--- | :--- | :--- |
| 1 | **Grzenda et al. (2020)** – Nghiên cứu đánh giá luồng dữ liệu có nhãn đến trễ | Phân tích vấn đề nhãn thực tế không xuất hiện ngay sau khi dữ liệu được dự đoán; việc đánh giá có thể cần xem xét thời điểm nhãn đến và các lần đánh giá lại. | Giúp xác định ảnh hưởng của độ trễ nhãn đến quá trình đánh giá mô hình và quyết định cập nhật. |
| 2 | **Bayram, Ahmed & Kassler (2022)** – From Concept Drift to Model Degradation: An Overview on Performance-Aware Drift Detectors | Tổng hợp, phân loại các dạng Concept Drift và phương pháp phát hiện drift dựa trên sự suy giảm hiệu suất mô hình. | Hỗ trợ lựa chọn cách theo dõi hiệu suất và phân biệt giữa phát hiện thay đổi dữ liệu với phát hiện suy giảm mô hình. |
| 3 | **Cerqueira, Torgo & Mozetič (2020)** – Evaluating Time Series Forecasting Models: An Empirical Study on Performance Estimation Methods | So sánh các phương pháp đánh giá dự báo chuỗi thời gian, bao gồm cross-validation, out-of-sample và prequential. | Là cơ sở lựa chọn cách chia dữ liệu theo thời gian và đánh giá mô hình dự báo PM2.5, tránh làm sai lệch thứ tự thời gian. |
| 4 | **Demšar (2006)** – Statistical Comparisons of Classifiers over Multiple Data Sets | Trình bày các phương pháp kiểm định thống kê để so sánh thuật toán học máy trên nhiều bộ dữ liệu, trong đó có Wilcoxon và Friedman. | Hỗ trợ xây dựng phương pháp so sánh kết quả thực nghiệm có cơ sở thống kê. |
| 5 | **Souza et al. (2020)** – Nghiên cứu về benchmark dữ liệu luồng và kho dữ liệu stream | Nhấn mạnh những vấn đề khi sử dụng dữ liệu luồng thực tế, bao gồm tính tuần tự, sự phụ thuộc thời gian và đặc điểm của từng bộ dữ liệu. | Gợi ý kiểm tra tính phù hợp của Elec2, đặc biệt là thứ tự thời gian, nhãn và baseline trước khi đánh giá drift. |

---

## 5. So sánh và phân tích phản biện

| Tiêu chí | Nghiên cứu When to Retrain (2026) | Nghiên cứu CARA (2024) | Nhận xét tổng hợp |
| :--- | :--- | :--- | :--- |
| **Trọng tâm** | So sánh các chính sách retraining trong điều kiện drift, ngân sách và độ trễ. | Tối ưu quyết định Retrain hoặc Keep dựa trên chi phí. | Hai nghiên cứu tiếp cận vấn đề từ hai góc độ bổ sung cho nhau. |
| **Cơ chế quyết định** | Lịch định kỳ, ngưỡng lỗi và bộ phát hiện ADWIN. | Chi phí lỗi thời của mô hình và chi phí retraining. | Có thể kết hợp tư duy phát hiện thay đổi với đánh giá lợi ích cập nhật. |
| **Yếu tố tài nguyên** | Nhấn mạnh ngân sách retraining và độ trễ. | Nhấn mạnh chi phí mô hình lỗi thời và tài nguyên retraining. | Cần xem xét cả chi phí tính toán lẫn hiệu quả dự đoán. |
| **Khả năng thích ứng** | So sánh thêm Incremental Learning và các chính sách retraining. | Đưa ra quyết định cập nhật dựa trên dữ liệu, truy vấn và chi phí. | Cần xác định phương pháp phù hợp với từng loại mô hình và dữ liệu. |
| **Hạn chế** | Có giới hạn về mô hình, giả định nhãn đến ngay và thiết lập thực nghiệm. | Chi phí tính toán của phép đo và giả định về chi phí retraining. | Cần kiểm chứng trong pipeline và bộ dữ liệu của dự án. |

### Nhận định tổng kết từ các nghiên cứu:
- **Phát hiện Concept Drift không đồng nghĩa với việc phải retraining ngay lập tức:** Một thay đổi trong dữ liệu chỉ nên dẫn đến cập nhật khi có bằng chứng cho thấy việc cập nhật mang lại lợi ích phù hợp.
- **Đánh giá đa chiều:** Chính sách retraining cần được đánh giá đồng thời theo hiệu suất dự đoán, số lần cập nhật, thời gian huấn luyện và tài nguyên sử dụng.
- **Tính chất dữ liệu và mô hình:** Đặc điểm của dữ liệu và mô hình có ảnh hưởng lớn đến hiệu quả của phương pháp cập nhật.
- **Thứ tự thời gian:** Với dữ liệu chuỗi thời gian, việc chia tập và đánh giá phải bảo toàn thứ tự thời gian.
- **Độ trễ nhãn:** Độ trễ nhãn có thể ảnh hưởng đến thời điểm phát hiện suy giảm hiệu suất và khả năng đánh giá mô hình.

---

## 6. Khoảng trống nghiên cứu

Từ việc tổng hợp các nghiên cứu, có thể xác định một số vấn đề cần tiếp tục xem xét trong phạm vi đề tài:
1. **Thiếu đánh giá đồng thời hiệu suất và chi phí:** Nhiều phương pháp tập trung vào khả năng phát hiện drift hoặc cải thiện độ chính xác, trong khi chi phí retraining và độ trễ triển khai cũng ảnh hưởng trực tiếp đến hiệu quả hệ thống.
2. **Phụ thuộc vào đặc điểm dữ liệu:** Một chính sách có thể hoạt động tốt với dạng drift này nhưng không hiệu quả với dạng drift khác. Vì vậy, cần đánh giá theo từng bộ dữ liệu và nhiệm vụ.
3. **Vấn đề nhãn đến trễ:** Trong môi trường thực tế, nhãn có thể không xuất hiện ngay sau khi dự đoán, làm ảnh hưởng đến việc giám sát hiệu suất và ra quyết định cập nhật.
4. **Khác biệt giữa phát hiện drift và quyết định retraining:** Phát hiện sự thay đổi không đồng nghĩa với việc mô hình đã suy giảm đến mức cần huấn luyện lại.
5. **Khả năng tích hợp vào pipeline tự động:** Quyết định cập nhật cần được kết nối với quá trình tiếp nhận dữ liệu, đánh giá mô hình, quản lý phiên bản và triển khai.

*(Các khoảng trống trên là định hướng nghiên cứu được rút ra từ tài liệu, không phải khẳng định rằng các nghiên cứu trước hoàn toàn chưa giải quyết những vấn đề này).*

---

## 7. Định hướng nghiên cứu đề xuất

Dựa trên cơ sở lý thuyết và các nghiên cứu đã tổng hợp, đề tài đề xuất tập trung vào việc đánh giá các phương pháp cập nhật thích ứng trong quy trình dữ liệu tự động.

| Nhánh nghiên cứu | Dữ liệu / Nhiệm vụ | Mô hình và phương pháp đang được yêu cầu xác nhận | Hướng nghiên cứu đề xuất |
| :--- | :--- | :--- | :--- |
| **ELEC2** | Phân loại trên dữ liệu luồng | GaussianNB, Hoeffding Tree; ADWIN/DDM; Prequential Evaluation | Đánh giá khả năng phát hiện drift và duy trì hiệu suất mô hình khi dữ liệu thay đổi. |
| **Beijing** | Dự báo PM2.5 trước 1 giờ | Persistence, LightGBM, Random Forest | So sánh các mô hình dự báo và đánh giá khả năng duy trì hiệu suất theo thời gian. |
| **Vietnam** | Dự báo chất lượng không khí | Persistence, LightGBM, River Linear Regression | Nghiên cứu khả năng cập nhật mô hình trên dữ liệu thực tế theo thời gian. |

*(Các mô hình và phương pháp trong bảng là nội dung AI Team yêu cầu xác nhận trước khi phê duyệt, chưa được xem là danh sách đã được phê duyệt cuối cùng).*

### Định hướng tổng thể:
- Theo dõi sự thay đổi của dữ liệu và hiệu suất mô hình.
- So sánh các chính sách cập nhật như Periodic Retraining, Error-threshold Retraining, Drift-triggered Retraining và No-retrain.
- Xem xét khả năng sử dụng Incremental Learning khi mô hình và giao diện triển khai hỗ trợ.
- Đánh giá tác động của chi phí retraining, độ trễ và thời điểm nhãn đến.
- Xây dựng quy trình đánh giá mô hình mới trước khi thay thế mô hình hiện tại.

---

## 8. Thiết kế thực nghiệm sơ bộ

*(Thiết kế dưới đây là đề xuất ban đầu để định hướng thực nghiệm. Các tham số, ngưỡng, dữ liệu đầu vào và tiêu chí triển khai cần được thống nhất với Data Pipeline, AI và MLOps trước khi thực hiện).*

### 8.1. Mục tiêu thực nghiệm
- Đánh giá khả năng duy trì hiệu suất của mô hình khi dữ liệu thay đổi.
- So sánh hiệu quả giữa các chính sách cập nhật.
- Đo lường sự đánh đổi giữa hiệu suất dự đoán và chi phí retraining.
- Đánh giá khả năng tích hợp chính sách cập nhật vào pipeline tự động.

### 8.2. Các phương pháp so sánh
| Phương pháp | Cơ chế | Mục đích |
| :--- | :--- | :--- |
| **No-retrain** | Giữ nguyên mô hình, không huấn luyện lại trong quá trình thử nghiệm. | Baseline để đánh giá mức suy giảm nếu không cập nhật. |
| **Periodic Retraining** | Huấn luyện lại theo chu kỳ xác định. | Đánh giá chính sách cập nhật định kỳ. |
| **Error-threshold Retraining** | Kích hoạt retraining khi chỉ số lỗi vượt ngưỡng. | Đánh giá chính sách dựa trên suy giảm hiệu suất. |
| **Drift-triggered Retraining** | Kích hoạt retraining khi bộ phát hiện drift đưa ra cảnh báo. | Đánh giá chính sách dựa trên tín hiệu phát hiện drift. |
| **Incremental Learning (nếu hỗ trợ)** | Cập nhật mô hình từng phần khi dữ liệu mới và nhãn tương ứng sẵn sàng. | Đánh giá khả năng thích ứng mà không cần retraining toàn phần thường xuyên. |

### 8.3. Quy trình thực nghiệm dự kiến
1. Tiếp nhận dữ liệu theo thứ tự thời gian từ Data Pipeline.
2. Kiểm tra dữ liệu, kiểu dữ liệu, feature, target và nguy cơ data leakage.
3. Huấn luyện mô hình ban đầu trên tập dữ liệu huấn luyện.
4. Đưa dữ liệu mới vào hệ thống và thực hiện dự đoán.
5. Theo dõi hiệu suất hoặc tín hiệu drift theo phương pháp đã lựa chọn.
6. Kích hoạt chính sách cập nhật tương ứng khi thỏa mãn điều kiện.
7. Huấn luyện hoặc cập nhật mô hình bằng dữ liệu phù hợp.
8. Đánh giá mô hình mới trước khi thay thế mô hình đang sử dụng.
9. Ghi nhận kết quả, thời gian xử lý và thông tin phiên bản mô hình.

### 8.4. Chỉ số đánh giá
| Nhóm đánh giá | Chỉ số đề xuất | Ý nghĩa |
| :--- | :--- | :--- |
| **Hiệu suất phân loại** | Accuracy, F1-score | Đánh giá chất lượng dự đoán trên ELEC2. |
| **Hiệu suất hồi quy** | MAE, RMSE | Đánh giá sai số dự báo PM2.5. |
| **Phát hiện drift** | Detection Delay, số lần cảnh báo | Đánh giá khả năng phát hiện và thời điểm phát hiện thay đổi. |
| **Chi phí cập nhật** | Retraining Count, Retraining Duration | Đánh giá tần suất và thời gian huấn luyện lại. |
| **Tài nguyên hệ thống** | CPU Usage, thời gian xử lý | Đánh giá chi phí vận hành nếu hệ thống có thể thu thập. |
| **Triển khai** | Deployment Latency, Model Version | Theo dõi độ trễ triển khai và lịch sử phiên bản mô hình. |

### 8.5. Phương pháp đánh giá và so sánh
- Sử dụng Prequential Evaluation cho nhiệm vụ phân loại dữ liệu luồng khi phù hợp.
- Với dự báo PM2.5, ưu tiên chia dữ liệu theo thời gian và đánh giá trên các giai đoạn tương lai; có thể sử dụng đánh giá rolling hoặc repeated holdout.
- Đảm bảo các chính sách được so sánh trên cùng điều kiện dữ liệu và cùng cách thiết lập thực nghiệm.
- Có thể sử dụng kiểm định thống kê phù hợp với thiết kế so sánh; Wilcoxon hoặc Friedman là các hướng tham khảo tùy số lượng phương pháp và bộ dữ liệu.
- Báo cáo cả độ lớn khác biệt và kết quả thống kê, không chỉ dựa vào một chỉ số Accuracy hoặc RMSE.

### 8.6. Các nội dung cần xác nhận liên nhóm
| Nội dung | Team liên quan | Vấn đề cần thống nhất |
| :--- | :--- | :--- |
| **Feature Contract** | Data Pipeline + AI | Feature, target, kiểu dữ liệu và quy tắc chống leakage theo từng dataset. |
| **Model Interface** | AI + MLOps | Đầu vào huấn luyện, đầu ra dự đoán và cách quản lý phiên bản mô hình. |
| **Evaluation** | AI | Accuracy/F1 cho phân loại; MAE/RMSE cho hồi quy và cách đánh giá theo thời gian. |
| **Drift Event** | AI + MLOps | Event schema, detector identifier và ý nghĩa timestamp. |
| **Retraining** | AI + MLOps | Điều kiện kích hoạt, số lượng dữ liệu tối thiểu và tiêu chí chấp nhận mô hình mới. |
| **Data Stream** | Data Pipeline + AI | Thứ tự thời gian, dữ liệu đến trễ và thời điểm nhãn có sẵn. |

---

## 9. Kết luận (Conclusion)

Các nghiên cứu được tổng hợp cho thấy cập nhật thích ứng mô hình học máy là một vấn đề có nhiều yếu tố liên quan, không chỉ giới hạn ở việc phát hiện Concept Drift. Việc lựa chọn thời điểm và phương pháp retraining cần cân nhắc hiệu suất dự đoán, chi phí tính toán, độ trễ, đặc điểm dữ liệu và khả năng cập nhật của mô hình.

Hai nghiên cứu chính cung cấp hai góc nhìn bổ sung:
- Nghiên cứu của **Dasari** tập trung so sánh các chính sách retraining trong điều kiện ngân sách và độ trễ.
- Nghiên cứu **CARA** tập trung cân bằng chi phí mô hình lỗi thời với chi phí retraining.

Các nghiên cứu bổ trợ cung cấp cơ sở cho đánh giá dữ liệu luồng, xử lý nhãn đến trễ, phát hiện drift và kiểm định kết quả thực nghiệm.

Trên cơ sở đó, đề tài định hướng nghiên cứu và so sánh các phương pháp cập nhật thích ứng trong ba nhánh dữ liệu **ELEC2**, **Beijing** và **Vietnam**. Mục tiêu là tìm hiểu cách duy trì hiệu suất mô hình trong môi trường dữ liệu thay đổi, đồng thời đảm bảo quá trình cập nhật có thể được đánh giá và tích hợp vào quy trình dữ liệu tự động.

---

## 10. Tài liệu tham khảo

1. **Dasari, S. (2026).** *When to Retrain: An Empirical Study of Retraining Policies for Streaming ML Under Concept Drift, Budget, and Latency Constraints.* arXiv:2608.19488.
2. **Mahadevan, A., & Mathioudakis, M. (2024).** *Cost-aware retraining for machine learning.* Knowledge-Based Systems, 293, 111610. https://doi.org/10.1016/j.knosys.2024.111610
3. **Bayram, F., Ahmed, B. S., & Kassler, A. (2022).** *From concept drift to model degradation: An overview on performance-aware drift detectors.* Knowledge-Based Systems, 245, 108632.
4. **Cerqueira, V., Torgo, L., & Mozetič, I. (2020).** *Evaluating time series forecasting models: An empirical study on performance estimation methods.* Machine Learning, 109, 1997–2028. https://doi.org/10.1007/s10994-020-05910-7
5. **Demšar, J. (2006).** *Statistical comparisons of classifiers over multiple data sets.* Journal of Machine Learning Research, 7, 1–30.
6. **Grzenda et al. (2020).** *Nghiên cứu về đánh giá mô hình trên luồng dữ liệu có nhãn đến trễ.*
7. **Souza et al. (2020).** *Nghiên cứu về benchmark và dữ liệu luồng.*
