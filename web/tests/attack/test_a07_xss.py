"""A07. Tiêm mã giao diện (XSS) qua tên file và kết quả.

Đọc `web/static/index.html` trước: mọi chỗ hiển thị dữ liệu người dùng gõ trực tiếp (tên tài xế, biển số)
hoặc tên file/chuyến đi đều dùng `.textContent`, không dùng `.innerHTML`. Những chỗ dùng `.innerHTML` chỉ
lắp dữ liệu có dạng cố định từ máy chủ (nhãn `NORMAL`/`AGGRESSIVE`/`DROWSY`, số, thời gian) -- không có
chuỗi tự do nào của người dùng đi qua đó. Test này xác nhận lại bằng TRÌNH DUYỆT THẬT (Selenium), không
chỉ đọc code: nếu một bản sửa sau này vô tình đổi `textContent` thành `innerHTML` ở ô tài xế/biển số,
test phải đỏ.
"""
import os
import time

import pytest

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.edge.service import Service as EdgeService
except ImportError:
    webdriver = None

pytestmark = pytest.mark.skipif(webdriver is None, reason='selenium chưa cài trong môi trường này')

PAYLOAD = '<img src=x onerror="window.__xss_fired=(window.__xss_fired||0)+1">'


@pytest.fixture
def browser():
    options = webdriver.EdgeOptions()
    options.add_argument('--headless=new')
    options.add_argument('--window-size=1280,900')
    drv = webdriver.Edge(options=options)
    try:
        yield drv
    finally:
        drv.quit()


def test_driver_and_plate_fields_are_rendered_as_literal_text(demo_run, browser, tmp_path):
    """Gõ payload XSS vào ô "tài xế" và "biển số", TẢI LÊN một chuyến hợp lệ nhỏ (ô tài xế/biển số chỉ
    được đọc ở nhánh tải lên, không phải nhánh "chuyến mẫu" -- xem `render()` trong index.html), rồi
    kiểm tra: (a) trang không hề chạy đoạn mã đó, (b) chữ hiện ra trên trang là nguyên văn (có dấu `<`,
    `>`), không bị trình duyệt diễn giải thành thẻ HTML."""
    import attack_kit as kit
    trip_file = tmp_path / 'RAW_ACCELEROMETERS.txt'
    trip_file.write_bytes(kit.sensor_text(kit.trip_with(lambda t, rng: rng.normal(0, 0.05, (len(t), 3)))))

    c = demo_run['container']
    browser.get(c.url + '/')
    browser.execute_script('window.__xss_fired = 0;')
    browser.find_element(By.ID, 'driver').send_keys(PAYLOAD)
    browser.find_element(By.ID, 'plate').send_keys(PAYLOAD)
    browser.find_element(By.ID, 'pickAcc').send_keys(str(trip_file))
    browser.find_element(By.ID, 'runBtn').click()

    deadline = time.time() + 90
    while time.time() < deadline:
        if browser.execute_script("return !document.getElementById('resultPanel').hidden"):
            break
        time.sleep(1)
    else:
        pytest.fail('trang không hiện kết quả trong 90s (demo không chạy xong?)')

    fired = browser.execute_script('return window.__xss_fired;')
    assert not fired, 'mã trong payload ĐÃ chạy được (window.__xss_fired > 0) -- lỗ hổng XSS thật'

    driver_text = browser.find_element(By.ID, 'iDriver').text
    plate_text = browser.find_element(By.ID, 'iPlate').text
    assert PAYLOAD in driver_text, f'ô tài xế không hiện nguyên văn payload (có thể đã bị lọc hoặc chạy): {driver_text!r}'
    assert PAYLOAD in plate_text, f'ô biển số không hiện nguyên văn payload: {plate_text!r}'

    # Không có <img> nào được DOM thực sự tạo ra từ payload (nếu có, nghĩa là bị hiểu thành HTML).
    img_count = browser.execute_script(
        "return document.querySelectorAll('#iDriver img, #iPlate img').length;")
    assert img_count == 0, f'tìm thấy {img_count} thẻ <img> thật trong vùng hiển thị -- payload đã bị diễn giải thành HTML'
