from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import json
import csv
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class VehicleCounter:
    def __init__(self, video_source=None):
        self.video_source = os.getenv('VIDEO_SOURCE')
        if not self.video_source:
            raise ValueError("VIDEO_SOURCE not found in .env file")
        self.model = YOLO('yolov8n.pt')
        self.tracker = defaultdict(dict)
        self.vehicle_count = {'entering': 0, 'leaving': 0}
        self.counted_ids = set()
        self.line_position = 0.5
        self.line_offset = 20
        
        # Create output directory if it doesn't exist
        self.output_dir = 'vehicle_counts'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize count files
        self.init_count_files()
        
    def init_count_files(self):
        """Initialize count files with headers"""
        # CSV file for detailed logs
        self.csv_path = os.path.join(self.output_dir, 'vehicle_counts.csv')
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Entering', 'Leaving', 'Total'])
        
        # JSON file for current totals
        self.json_path = os.path.join(self.output_dir, 'current_totals.json')
        self.save_json_counts()
    
    def save_counts(self):
        """Save counts to both CSV and JSON files"""
        # Save to CSV
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total = self.vehicle_count['entering'] + self.vehicle_count['leaving']
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                self.vehicle_count['entering'],
                self.vehicle_count['leaving'],
                total
            ])
        
        # Save to JSON
        self.save_json_counts()
    
    def save_json_counts(self):
        """Save current counts to JSON file"""
        data = {
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'counts': self.vehicle_count,
            'total': self.vehicle_count['entering'] + self.vehicle_count['leaving']
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_source)
            frame_count = 0  # To track when to save counts
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Failed to receive frame. Attempting to reconnect...")
                    cap.release()
                    cap = cv2.VideoCapture(self.video_source)
                    continue
                
                height, width = frame.shape[:2]
                counting_line_y = int(height * self.line_position)
                
                # Draw counting line
                cv2.line(frame, (0, counting_line_y), (width, counting_line_y), 
                        (0, 255, 0), 2)
                
                # Run YOLOv8 tracking
                results = self.model.track(frame, persist=True, conf=0.5, 
                                        classes=[2, 3, 5, 7])
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy()
                    
                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = box.astype(int)
                        center_y = (y1 + y2) // 2
                        track_id = int(track_id)
                        
                        if track_id in self.tracker:
                            prev_center_y = self.tracker[track_id]['center_y']
                            
                            if track_id not in self.counted_ids:
                                if (prev_center_y < counting_line_y and 
                                    center_y >= counting_line_y):
                                    self.vehicle_count['entering'] += 1
                                    self.counted_ids.add(track_id)
                                elif (prev_center_y > counting_line_y and 
                                      center_y <= counting_line_y):
                                    self.vehicle_count['leaving'] += 1
                                    self.counted_ids.add(track_id)
                        
                        self.tracker[track_id] = {'center_y': center_y}
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Display counts
                cv2.putText(frame, f"Entering: {self.vehicle_count['entering']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Leaving: {self.vehicle_count['leaving']}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Vehicle Counter', frame)
                
                # Save counts every 30 frames (approximately every second)
                frame_count += 1
                if frame_count % 30 == 0:
                    self.save_counts()
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            # Save final counts before closing
            self.save_counts()
            cap.release()
            cv2.destroyAllWindows()

# Usage
counter = VehicleCounter()
counter.run()