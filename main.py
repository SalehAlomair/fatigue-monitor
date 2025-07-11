import cv2
import dlib
import threading
from imutils import face_utils
from scipy.spatial import distance as dist
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import simpleaudio as sa
import time


class FatigueMonitorApp:
    """Modern Fatigue Monitor with Enhanced GUI"""

    def __init__(self, root, camera_index: int = 0):
        self.root = root
        self.root.title("Fatigue Monitor Pro")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Modern color scheme
        self.colors = {
            'bg': '#1a1a1a',
            'surface': '#2d2d2d',
            'card': '#3d3d3d',
            'accent': '#4a9eff',
            'success': '#00c896',
            'warning': '#ff9500',
            'danger': '#ff5757',
            'text': '#ffffff',
            'text_secondary': '#a0a0a0',
            'border': '#4a4a4a'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Style configuration
        self.setup_styles()
        
        self.camera_index = camera_index
        self.cap = None
        self.monitoring = False
        self.alarm_wav = "alarm.wav"
        self.start_time = None

        # Detection models setup
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # EAR constants
        self.EAR_THRESHOLD = 0.25
        self.CONSEC_FRAMES = 20
        self.counter = 0
        self.alerts = 0
        self.alarm_on = False
        self.total_frames = 0
        self.drowsy_frames = 0
        self.blink_count = 0
        self.avg_ear = 0.0

        self.setup_ui()

    def setup_styles(self):
        """Configure modern styles for ttk widgets"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Modern.TFrame', background=self.colors['surface'])
        style.configure('Card.TFrame', background=self.colors['card'], relief='flat')
        style.configure('Modern.TLabel', background=self.colors['surface'], foreground=self.colors['text'])
        style.configure('Card.TLabel', background=self.colors['card'], foreground=self.colors['text'])
        style.configure('Title.TLabel', background=self.colors['surface'], foreground=self.colors['text'], font=('Segoe UI', 16, 'bold'))
        style.configure('Subtitle.TLabel', background=self.colors['card'], foreground=self.colors['text_secondary'], font=('Segoe UI', 12))
        
        # Progress bar styles
        style.configure('Success.Horizontal.TProgressbar', background=self.colors['success'], troughcolor=self.colors['surface'])
        style.configure('Warning.Horizontal.TProgressbar', background=self.colors['warning'], troughcolor=self.colors['surface'])
        style.configure('Danger.Horizontal.TProgressbar', background=self.colors['danger'], troughcolor=self.colors['surface'])

    def setup_ui(self):
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill='both', expand=True, padx=20, pady=20)

        # Header
        self.create_header(main_container)
        
        # Content area
        content_frame = tk.Frame(main_container, bg=self.colors['bg'])
        content_frame.pack(fill='both', expand=True, pady=(20, 0))
        
        # Left panel (video + controls)
        left_panel = tk.Frame(content_frame, bg=self.colors['bg'])
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Right panel (statistics)
        right_panel = tk.Frame(content_frame, bg=self.colors['bg'], width=350)
        right_panel.pack(side='right', fill='y', padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Setup panels
        self.create_video_panel(left_panel)
        self.create_control_panel(left_panel)
        self.create_stats_panel(right_panel)
        
        # Bottom status bar
        self.create_status_bar(main_container)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_time()

    def create_header(self, parent):
        """Create modern header with gradient effect"""
        header_frame = tk.Frame(parent, bg=self.colors['accent'], height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        # Title and subtitle
        title_container = tk.Frame(header_frame, bg=self.colors['accent'])
        title_container.pack(expand=True)
        
        title_label = tk.Label(title_container, text="🚗 Fatigue Monitor Pro", 
                              font=('Segoe UI', 24, 'bold'), fg='white', bg=self.colors['accent'])
        title_label.pack(pady=(15, 0))
        
        subtitle_label = tk.Label(title_container, text="AI-Powered Driver Drowsiness Detection System", 
                                 font=('Segoe UI', 11), fg='white', bg=self.colors['accent'])
        subtitle_label.pack(pady=(0, 15))

    def create_video_panel(self, parent):
        """Create video display panel"""
        video_frame = tk.Frame(parent, bg=self.colors['surface'], relief='flat', bd=2)
        video_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        # Video header
        video_header = tk.Frame(video_frame, bg=self.colors['surface'], height=40)
        video_header.pack(fill='x', padx=10, pady=(10, 0))
        video_header.pack_propagate(False)
        
        tk.Label(video_header, text="📹 Live Camera Feed", 
                font=('Segoe UI', 14, 'bold'), fg=self.colors['text'], bg=self.colors['surface']).pack(side='left', pady=10)
        
        # Video display
        self.video_label = tk.Label(video_frame, bg='black', text="Camera Feed\nWill Appear Here", 
                                   fg='white', font=('Segoe UI', 16), relief='flat', bd=2)
        self.video_label.pack(fill='both', expand=True, padx=10, pady=(0, 10))

    def create_control_panel(self, parent):
        """Create modern control panel"""
        control_frame = tk.Frame(parent, bg=self.colors['surface'], relief='flat', bd=2)
        control_frame.pack(fill='x', pady=(0, 20))
        
        # Control header
        control_header = tk.Frame(control_frame, bg=self.colors['surface'], height=40)
        control_header.pack(fill='x', padx=10, pady=(10, 0))
        control_header.pack_propagate(False)
        
        tk.Label(control_header, text="🎮 Control Panel", 
                font=('Segoe UI', 14, 'bold'), fg=self.colors['text'], bg=self.colors['surface']).pack(side='left', pady=10)
        
        # Buttons container
        buttons_container = tk.Frame(control_frame, bg=self.colors['surface'])
        buttons_container.pack(fill='x', padx=20, pady=(0, 20))
        
        # Primary buttons
        btn_frame = tk.Frame(buttons_container, bg=self.colors['surface'])
        btn_frame.pack(fill='x', pady=10)
        
        self.start_btn = tk.Button(btn_frame, text="▶️ Start Monitoring", 
                                  command=self.start_monitoring, bg=self.colors['success'], fg='white',
                                  font=('Segoe UI', 12, 'bold'), relief='flat', bd=0,
                                  padx=20, pady=12, cursor='hand2')
        self.start_btn.pack(side='left', padx=(0, 10))
        
        self.stop_btn = tk.Button(btn_frame, text="⏹️ Stop Monitoring", 
                                 command=self.stop_monitoring, bg=self.colors['danger'], fg='white',
                                 font=('Segoe UI', 12, 'bold'), relief='flat', bd=0,
                                 padx=20, pady=12, state="disabled", cursor='hand2')
        self.stop_btn.pack(side='left', padx=10)
        
        self.settings_btn = tk.Button(btn_frame, text="⚙️ Settings", 
                                     command=self.open_settings, bg=self.colors['accent'], fg='white',
                                     font=('Segoe UI', 12, 'bold'), relief='flat', bd=0,
                                     padx=20, pady=12, cursor='hand2')
        self.settings_btn.pack(side='left', padx=10)
        
        # Quick stats in control panel
        quick_stats = tk.Frame(buttons_container, bg=self.colors['surface'])
        quick_stats.pack(fill='x', pady=(10, 0))
        
        self.session_status = tk.Label(quick_stats, text="⏸️ Session Status: Idle", 
                                      font=('Segoe UI', 11), fg=self.colors['text_secondary'], bg=self.colors['surface'])
        self.session_status.pack(side='left')
        
        self.camera_status = tk.Label(quick_stats, text="📷 Camera: Disconnected", 
                                     font=('Segoe UI', 11), fg=self.colors['text_secondary'], bg=self.colors['surface'])
        self.camera_status.pack(side='right')

    def create_stats_panel(self, parent):
        """Create comprehensive statistics panel"""
        # Main stats container
        stats_container = tk.Frame(parent, bg=self.colors['bg'])
        stats_container.pack(fill='both', expand=True)
        
        # Stats header
        stats_header = tk.Frame(stats_container, bg=self.colors['surface'], height=50)
        stats_header.pack(fill='x', pady=(0, 10))
        stats_header.pack_propagate(False)
        
        tk.Label(stats_header, text="📊 Real-time Statistics", 
                font=('Segoe UI', 14, 'bold'), fg=self.colors['text'], bg=self.colors['surface']).pack(expand=True)
        
        # Time and session info
        self.create_time_card(stats_container)
        
        # EAR monitoring card
        self.create_ear_card(stats_container)
        
        # Alerts card
        self.create_alerts_card(stats_container)
        
        # Performance metrics
        self.create_performance_card(stats_container)

    def create_time_card(self, parent):
        """Create time tracking card"""
        time_card = tk.Frame(parent, bg=self.colors['card'], relief='flat', bd=1)
        time_card.pack(fill='x', pady=(0, 10))
        
        tk.Label(time_card, text="⏱️ Session Time", 
                font=('Segoe UI', 12, 'bold'), fg=self.colors['text'], bg=self.colors['card']).pack(anchor='w', padx=15, pady=(10, 5))
        
        self.time_var = tk.StringVar(value="00:00:00")
        time_display = tk.Label(time_card, textvariable=self.time_var, 
                               font=('Segoe UI', 20, 'bold'), fg=self.colors['accent'], bg=self.colors['card'])
        time_display.pack(anchor='w', padx=15, pady=(0, 10))

    def create_ear_card(self, parent):
        """Create EAR monitoring card"""
        ear_card = tk.Frame(parent, bg=self.colors['card'], relief='flat', bd=1)
        ear_card.pack(fill='x', pady=(0, 10))
        
        tk.Label(ear_card, text="👁️ Eye Aspect Ratio", 
                font=('Segoe UI', 12, 'bold'), fg=self.colors['text'], bg=self.colors['card']).pack(anchor='w', padx=15, pady=(10, 5))
        
        ear_display_frame = tk.Frame(ear_card, bg=self.colors['card'])
        ear_display_frame.pack(fill='x', padx=15, pady=(0, 10))
        
        self.ear_var = tk.StringVar(value="--")
        ear_value = tk.Label(ear_display_frame, textvariable=self.ear_var, 
                            font=('Segoe UI', 18, 'bold'), fg=self.colors['success'], bg=self.colors['card'])
        ear_value.pack(side='left')
        
        # EAR threshold indicator
        threshold_label = tk.Label(ear_display_frame, text=f"Threshold: {self.EAR_THRESHOLD}", 
                                  font=('Segoe UI', 10), fg=self.colors['text_secondary'], bg=self.colors['card'])
        threshold_label.pack(side='right')
        
        # Drowsiness progress
        tk.Label(ear_card, text="Drowsiness Level", 
                font=('Segoe UI', 10), fg=self.colors['text_secondary'], bg=self.colors['card']).pack(anchor='w', padx=15)
        
        progress_frame = tk.Frame(ear_card, bg=self.colors['card'])
        progress_frame.pack(fill='x', padx=15, pady=(5, 10))
        
        self.drowsy_progress = ttk.Progressbar(progress_frame, mode='determinate', 
                                              length=300, style='Danger.Horizontal.TProgressbar')
        self.drowsy_progress.pack(fill='x')

    def create_alerts_card(self, parent):
        """Create alerts monitoring card"""
        alerts_card = tk.Frame(parent, bg=self.colors['card'], relief='flat', bd=1)
        alerts_card.pack(fill='x', pady=(0, 10))
        
        tk.Label(alerts_card, text="🚨 Alert System", 
                font=('Segoe UI', 12, 'bold'), fg=self.colors['text'], bg=self.colors['card']).pack(anchor='w', padx=15, pady=(10, 5))
        
        alerts_info = tk.Frame(alerts_card, bg=self.colors['card'])
        alerts_info.pack(fill='x', padx=15, pady=(0, 10))
        
        self.alert_var = tk.StringVar(value="0")
        alert_value = tk.Label(alerts_info, textvariable=self.alert_var, 
                              font=('Segoe UI', 18, 'bold'), fg=self.colors['danger'], bg=self.colors['card'])
        alert_value.pack(side='left')
        
        tk.Label(alerts_info, text="Total Alerts", 
                font=('Segoe UI', 10), fg=self.colors['text_secondary'], bg=self.colors['card']).pack(side='left', padx=(10, 0))
        
        # Blink counter
        blink_frame = tk.Frame(alerts_card, bg=self.colors['card'])
        blink_frame.pack(fill='x', padx=15, pady=(0, 10))
        
        self.blink_var = tk.StringVar(value="0")
        blink_value = tk.Label(blink_frame, textvariable=self.blink_var, 
                              font=('Segoe UI', 14, 'bold'), fg=self.colors['text'], bg=self.colors['card'])
        blink_value.pack(side='left')
        
        tk.Label(blink_frame, text="Blinks", 
                font=('Segoe UI', 10), fg=self.colors['text_secondary'], bg=self.colors['card']).pack(side='left', padx=(10, 0))

    def create_performance_card(self, parent):
        """Create performance metrics card"""
        perf_card = tk.Frame(parent, bg=self.colors['card'], relief='flat', bd=1)
        perf_card.pack(fill='x', pady=(0, 10))
        
        tk.Label(perf_card, text="⚡ Performance Metrics", 
                font=('Segoe UI', 12, 'bold'), fg=self.colors['text'], bg=self.colors['card']).pack(anchor='w', padx=15, pady=(10, 5))
        
        # FPS counter
        fps_frame = tk.Frame(perf_card, bg=self.colors['card'])
        fps_frame.pack(fill='x', padx=15, pady=(0, 5))
        
        self.fps_var = tk.StringVar(value="0")
        tk.Label(fps_frame, text="FPS:", font=('Segoe UI', 10), fg=self.colors['text_secondary'], bg=self.colors['card']).pack(side='left')
        fps_value = tk.Label(fps_frame, textvariable=self.fps_var, 
                            font=('Segoe UI', 10, 'bold'), fg=self.colors['success'], bg=self.colors['card'])
        fps_value.pack(side='left', padx=(5, 0))
        
        # Drowsiness percentage
        drowsy_frame = tk.Frame(perf_card, bg=self.colors['card'])
        drowsy_frame.pack(fill='x', padx=15, pady=(0, 10))
        
        self.drowsy_percent_var = tk.StringVar(value="0.0%")
        tk.Label(drowsy_frame, text="Drowsiness:", font=('Segoe UI', 10), fg=self.colors['text_secondary'], bg=self.colors['card']).pack(side='left')
        drowsy_value = tk.Label(drowsy_frame, textvariable=self.drowsy_percent_var, 
                               font=('Segoe UI', 10, 'bold'), fg=self.colors['warning'], bg=self.colors['card'])
        drowsy_value.pack(side='left', padx=(5, 0))

    def create_status_bar(self, parent):
        """Create bottom status bar"""
        status_bar = tk.Frame(parent, bg=self.colors['surface'], height=30)
        status_bar.pack(fill='x', pady=(20, 0))
        status_bar.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="Ready to start monitoring")
        status_label = tk.Label(status_bar, textvariable=self.status_var, 
                               font=('Segoe UI', 10), fg=self.colors['text_secondary'], bg=self.colors['surface'])
        status_label.pack(side='left', padx=10, pady=5)
        
        # Version info
        version_label = tk.Label(status_bar, text="v2.0.0", 
                                font=('Segoe UI', 10), fg=self.colors['text_secondary'], bg=self.colors['surface'])
        version_label.pack(side='right', padx=10, pady=5)

    def open_settings(self):
        """Open modern settings window"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings - Fatigue Monitor Pro")
        settings_window.geometry("500x600")
        settings_window.configure(bg=self.colors['bg'])
        settings_window.resizable(False, False)
        settings_window.grab_set()
        
        # Settings header
        header = tk.Frame(settings_window, bg=self.colors['accent'], height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text="⚙️ Settings", font=('Segoe UI', 16, 'bold'), 
                fg='white', bg=self.colors['accent']).pack(expand=True)
        
        # Settings content
        content = tk.Frame(settings_window, bg=self.colors['bg'])
        content.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Detection settings
        detection_frame = tk.Frame(content, bg=self.colors['card'], relief='flat', bd=1)
        detection_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(detection_frame, text="🎯 Detection Settings", 
                font=('Segoe UI', 12, 'bold'), fg=self.colors['text'], bg=self.colors['card']).pack(anchor='w', padx=15, pady=(10, 5))
        
        # EAR threshold
        ear_frame = tk.Frame(detection_frame, bg=self.colors['card'])
        ear_frame.pack(fill='x', padx=15, pady=5)
        
        tk.Label(ear_frame, text="EAR Threshold:", font=('Segoe UI', 10), 
                fg=self.colors['text'], bg=self.colors['card']).pack(anchor='w')
        ear_scale = tk.Scale(ear_frame, from_=0.1, to=0.4, resolution=0.01, 
                            orient='horizontal', bg=self.colors['card'], fg=self.colors['text'],
                            highlightthickness=0, troughcolor=self.colors['surface'])
        ear_scale.set(self.EAR_THRESHOLD)
        ear_scale.pack(fill='x', pady=(5, 10))
        
        # Consecutive frames
        frames_frame = tk.Frame(detection_frame, bg=self.colors['card'])
        frames_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        tk.Label(frames_frame, text="Consecutive Frames:", font=('Segoe UI', 10), 
                fg=self.colors['text'], bg=self.colors['card']).pack(anchor='w')
        frames_scale = tk.Scale(frames_frame, from_=10, to=50, 
                               orient='horizontal', bg=self.colors['card'], fg=self.colors['text'],
                               highlightthickness=0, troughcolor=self.colors['surface'])
        frames_scale.set(self.CONSEC_FRAMES)
        frames_scale.pack(fill='x', pady=(5, 10))
        
        # Camera settings
        camera_frame = tk.Frame(content, bg=self.colors['card'], relief='flat', bd=1)
        camera_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(camera_frame, text="📷 Camera Settings", 
                font=('Segoe UI', 12, 'bold'), fg=self.colors['text'], bg=self.colors['card']).pack(anchor='w', padx=15, pady=(10, 5))
        
        # Camera index
        cam_frame = tk.Frame(camera_frame, bg=self.colors['card'])
        cam_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        tk.Label(cam_frame, text="Camera Index:", font=('Segoe UI', 10), 
                fg=self.colors['text'], bg=self.colors['card']).pack(side='left')
        cam_var = tk.StringVar(value=str(self.camera_index))
        cam_entry = tk.Entry(cam_frame, textvariable=cam_var, font=('Segoe UI', 10), width=10)
        cam_entry.pack(side='right')
        
        # Buttons
        btn_frame = tk.Frame(content, bg=self.colors['bg'])
        btn_frame.pack(fill='x', pady=(20, 0))
        
        def apply_settings():
            try:
                self.EAR_THRESHOLD = ear_scale.get()
                self.CONSEC_FRAMES = frames_scale.get()
                self.camera_index = int(cam_var.get())
                messagebox.showinfo("Settings", "Settings applied successfully!")
                settings_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid camera index!")
        
        def reset_settings():
            ear_scale.set(0.25)
            frames_scale.set(20)
            cam_var.set("0")
        
        tk.Button(btn_frame, text="Apply", command=apply_settings, 
                 bg=self.colors['success'], fg='white', font=('Segoe UI', 11, 'bold'),
                 relief='flat', bd=0, padx=20, pady=8, cursor='hand2').pack(side='right', padx=(5, 0))
        
        tk.Button(btn_frame, text="Reset", command=reset_settings, 
                 bg=self.colors['warning'], fg='white', font=('Segoe UI', 11, 'bold'),
                 relief='flat', bd=0, padx=20, pady=8, cursor='hand2').pack(side='right', padx=5)
        
        tk.Button(btn_frame, text="Cancel", command=settings_window.destroy, 
                 bg=self.colors['danger'], fg='white', font=('Segoe UI', 11, 'bold'),
                 relief='flat', bd=0, padx=20, pady=8, cursor='hand2').pack(side='right', padx=5)

    def update_time(self):
        """Update session time and other metrics"""
        if self.monitoring and self.start_time:
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.time_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Update drowsiness percentage
            if self.total_frames > 0:
                drowsy_percent = (self.drowsy_frames / self.total_frames) * 100
                self.drowsy_percent_var.set(f"{drowsy_percent:.1f}%")
        
        self.root.after(1000, self.update_time)

    def start_monitoring(self):
        """Start the monitoring process"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot access camera!")
            return
        
        self.monitoring = True
        self.start_time = time.time()
        self.total_frames = 0
        self.drowsy_frames = 0
        self.blink_count = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        
        # Update UI
        self.start_btn["state"] = "disabled"
        self.stop_btn["state"] = "normal"
        self.settings_btn["state"] = "disabled"
        self.session_status.config(text="▶️ Session Status: Active", fg=self.colors['success'])
        self.camera_status.config(text="📷 Camera: Connected", fg=self.colors['success'])
        self.status_var.set("🟢 Monitoring active - AI is watching for drowsiness")
        
        self.video_loop()

    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.monitoring = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Reset UI
        self.video_label.config(image="", text="Camera Feed\nWill Appear Here", 
                               fg='white', bg='black')
        self.start_btn["state"] = "normal"
        self.stop_btn["state"] = "disabled"
        self.settings_btn["state"] = "normal"
        self.session_status.config(text="⏸️ Session Status: Idle", fg=self.colors['text_secondary'])
        self.camera_status.config(text="📷 Camera: Disconnected", fg=self.colors['text_secondary'])
        self.status_var.set("Ready to start monitoring")
        self.ear_var.set("--")
        self.fps_var.set("0")
        self.drowsy_progress['value'] = 0
        self.counter = 0
        self.alarm_on = False

    def on_close(self):
        """Handle window close event"""
        self.stop_monitoring()
        self.root.destroy()

    def play_alarm(self):
        """Play alarm sound"""
        try:
            sa.WaveObject.from_wave_file(self.alarm_wav).play()
        except Exception as e:
            print("Alarm error:", e)

    @staticmethod
    def calculate_EAR(eye):
        """Calculate Eye Aspect Ratio"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def video_loop(self):
        """Main video processing loop"""
        if not self.monitoring:
            return
        
        ret, frame = self.cap.read()
        if ret:
            self.total_frames += 1
            self.fps_counter += 1
            
            # Calculate FPS
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.fps_var.set(f"{self.fps_counter}")
                self.fps_counter = 0
                self.last_fps_time = current_time
            
            # Process frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)
            
            if faces:
                for face in faces:
                    shape = self.predictor(gray, face)
                    shape = face_utils.shape_to_np(shape)
                    left_eye = shape[self.lStart:self.lEnd]
                    right_eye = shape[self.rStart:self.rEnd]
                    ear = (self.calculate_EAR(left_eye) + self.calculate_EAR(right_eye)) / 2.0
                    
                    # Update EAR display
                    self.ear_var.set(f"{ear:.3f}")
                    
                    # Blink detection
                    if ear < self.EAR_THRESHOLD:
                        self.counter += 1
                        self.drowsy_frames += 1
                    else:
                        if self.counter > 0:
                            self.blink_count += 1
                            self.blink_var.set(f"{self.blink_count}")
                        self.counter = 0
                        self.alarm_on = False
                    
                    # Update progress bar
                    progress = min((self.counter / self.CONSEC_FRAMES) * 100, 100)
                    self.drowsy_progress['value'] = progress
                    
                    # Drowsiness detection
                    if self.counter >= self.CONSEC_FRAMES:
                        if not self.alarm_on:
                            self.alerts += 1
                            self.alert_var.set(f"{self.alerts}")
                            self.alarm_on = True
                            self.status_var.set("🚨 DROWSINESS DETECTED! Wake up!")
                            threading.Thread(target=self.play_alarm, daemon=True).start()
                        
                        # Draw warnings
                        cv2.putText(frame, "DROWSY!", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        cv2.putText(frame, "WAKE UP!", (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        if self.alarm_on:
                            self.status_var.set("🟢 Monitoring active - AI is watching for drowsiness")
                    
                    # Draw eye contours
                    cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
                    
                    # Draw face rectangle
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Draw EAR value on frame
                    cv2.putText(frame, f"EAR: {ear:.3f}", (10, frame.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                self.ear_var.set("No Face")
                self.drowsy_progress['value'] = 0
            
            # Update video display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            # Resize to fit the display area
            display_width = self.video_label.winfo_width()
            display_height = self.video_label.winfo_height()
            if display_width > 1 and display_height > 1:
                img = img.resize((display_width-4, display_height-4), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk, text="")

        self.root.after(10, self.video_loop)


if __name__ == "__main__":
    root = tk.Tk()
    app = FatigueMonitorApp(root, camera_index=0)
    root.mainloop()
