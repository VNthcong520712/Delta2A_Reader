import serial
import struct
import time
import sys
import matplotlib.pyplot as plt
import numpy as np

class Delta2ALiDAR:
	# --- HEX code ---
	START_BYTE = 0xAA
	DATA_TYPE_RPM_AND_MEAS = 0xAD
	DATA_TYPE_RPM_ONLY = 0xAE

	REFRESH_FACTOR = 60

	def __init__(self, serial_port, targetHz=None):
		self.port = serial_port
		self.targetHz = targetHz
		self.baudrate = 115200
		self.ser = None
		self.rx_buffer = bytearray()
		self.scan_points = []
		self.scan_ready_for_plot = False
		self.current_scan_frequency = 0.0

	def connect(self):
		try:
			self.ser = serial.Serial(self.port, self.baudrate, timeout=0.01)
			print(f"[PORT] Successfully connected to Delta2A at port {self.port}")
			return True
		except serial.SerialException as e:
			print(f"[PORT] Error while connecting to Delta2A at port {self.port}: {e}", file=sys.stderr)
			return False

	def disconnect(self):
		if self.ser and self.ser.is_open:
			self.ser.close()
			print("[LIDAR] Disconnected.")

	def send_freq(self, setTo = None):
		setTo = setTo if setTo is not None else self.targetHz
		if self.ser and self.ser.is_open:
			message = f"now:{self.current_scan_frequency:.2f}-to:{setTo:.2f}\n"
			self.ser.write(message.encode('utf-8'))

	def process_incoming_data(self):
		if self.ser and self.ser.in_waiting > 0:
			data_chunk = self.ser.read(self.ser.in_waiting)
			self.rx_buffer.extend(data_chunk)
		
		while True:
			start_index = self.rx_buffer.find(self.START_BYTE)
			if start_index == -1:
				self.rx_buffer.clear()
				return

			if start_index > 0:
				self.rx_buffer = self.rx_buffer[start_index:]

			if len(self.rx_buffer) < 8: return

			header = self.rx_buffer[1:8]
			packet_length, _, _, _, _ = struct.unpack('>HBBB H', header)
			full_packet_size = packet_length + 2

			if len(self.rx_buffer) < full_packet_size: return

			packet = self.rx_buffer[:full_packet_size]
			self.handle_packet(packet)
			self.rx_buffer = self.rx_buffer[full_packet_size:]

	def handle_packet(self, packet):
		data_type = packet[5]
		self.current_scan_frequency = packet[8] * 0.05
		if self.targetHz is not None: self.send_freq(self.current_scan_frequency)

		if data_type == self.DATA_TYPE_RPM_AND_MEAS:
			data_length, = struct.unpack('>H', packet[6:8])
			start_angle_deg = struct.unpack('>H', packet[11:13])[0] * 0.01
			sample_count = (data_length - 5) // 3
			if sample_count <= 0: return

			if start_angle_deg < 5.0 and len(self.scan_points) > 100:
				self.scan_ready_for_plot = True

			angle_increment = 360.0 / (16 * sample_count) if sample_count > 0 else 0
			for i in range(sample_count):
				offset = 13 + (i * 3)
				distance_mm = struct.unpack('>H', packet[offset+1:offset+3])[0] * 0.25
				angle_deg = start_angle_deg + (i * angle_increment)
				if distance_mm > 5: 
					self.scan_points.append((angle_deg, distance_mm))
		else: return
	
	def get_scan_data(self):
		if self.scan_ready_for_plot:
			self.scan_ready_for_plot = False
			data_to_return = list(self.scan_points)
			self.scan_points.clear()
			return data_to_return
		return None

def main():
	serial_port_name = 'COM15'

	lidar = Delta2ALiDAR(serial_port_name)
	if not lidar.connect():
		sys.exit("[SYSTEM] Failed to connect Delta2A")

	plt.ion()
	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(111, projection='polar')
	ax.set_theta_zero_location('S')
	ax.set_theta_direction(-1)
	ax.set_rlim(0, 5000)
	line, = ax.plot([], [], 'o', markersize=2, color='blue', alpha=0.75)
	title = ax.set_title("Waiting for Delta2A's data ...", va='bottom')
	plt.show(block=False)
	
	print("[SYSTEM] Start reading data")
	
	last_status_update_time = time.time()

	try:
		while True:
			lidar.process_incoming_data()
			scan_data = lidar.get_scan_data()
			
			if scan_data:
				angles_rad = np.deg2rad([p[0] for p in scan_data])
				distances_mm = [p[1] for p in scan_data]
				line.set_data(angles_rad, distances_mm)
				title.set_text(f"Processing (Freq: {lidar.current_scan_frequency:.2f} Hz)")
				print(f"[SYSTEM] Processing data. Current frequency: {lidar.current_scan_frequency:.2f} Hz")
				fig.canvas.draw_idle()
				
				last_status_update_time = time.time()
			else:
				current_time = time.time()
				if current_time - last_status_update_time > 1.0: 
					title.set_text(f"No data (Freq: {lidar.current_scan_frequency:.2f} Hz)")
					print(f"[SYSTEM] No data points. Current frequency: {lidar.current_scan_frequency:.2f} Hz")
					last_status_update_time = current_time 

			plt.pause(0.001)

	except KeyboardInterrupt:
		print("\n[SYSTEM] Stopping...")
	finally:
		lidar.disconnect()
		plt.ioff()
		print("[SYSTEM] All stop! Close.")

if __name__ == "__main__":
	main()