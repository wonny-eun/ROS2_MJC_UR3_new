import mujoco
import os

# 경로 설정 (본인의 환경에 맞게 자동 수정됨)
urdf_path = os.path.expanduser('~/ur3_control/src/Universal_Robots_ROS2_Description/urdf/ur3_clean.urdf')
xml_output_path = os.path.expanduser('~/ur3_control/src/Universal_Robots_ROS2_Description/urdf/ur3_robot.xml')

print(f"변환 시작: {urdf_path}")

try:
    # 1. URDF 읽어오기
    model = mujoco.MjModel.from_xml_path(urdf_path)
    
    # 2. MJCF(XML)로 저장
    mujoco.mj_saveLastXML(xml_output_path, model)
    
    print(f"성공! MJCF 파일이 생성되었습니다: {xml_output_path}")
    print("이제 이 XML 파일의 맨 아래에 <actuator> 섹션을 추가하세요.")
except Exception as e:
    print(f"에러 발생: {e}")
