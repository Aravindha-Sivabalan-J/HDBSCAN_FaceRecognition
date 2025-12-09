import cv2
import os
import sys
import numpy as np
import torch
import torch.nn as nn
try:
    import cuml
    from cuml.cluster import HDBSCAN as cumlHDBSCAN
    GPU_HDBSCAN_AVAILABLE = True
except ImportError:
    import hdbscan
    GPU_HDBSCAN_AVAILABLE = False

# --- GPU LIBRARY FIX ---
# Dynamically add nvidia-cudnn and nvidia-cublas paths to LD_LIBRARY_PATH
# This is required because pip-installed nvidia packages don't always register with ldconfig
try:
    import site
    site_packages = site.getsitepackages()
    
    # Try to find site-packages containing nvidia
    for path in site_packages:
        nvidia_path = os.path.join(path, "nvidia")
        if os.path.exists(nvidia_path):
            cudnn_lib = os.path.join(nvidia_path, "cudnn", "lib")
            cublas_lib = os.path.join(nvidia_path, "cublas", "lib")
            
            libs_to_add = []
            if os.path.exists(cudnn_lib): libs_to_add.append(cudnn_lib)
            if os.path.exists(cublas_lib): libs_to_add.append(cublas_lib)
            
            if libs_to_add:
                current_ld = os.environ.get("LD_LIBRARY_PATH", "")
                new_path = ":".join(libs_to_add) + ":" + current_ld
                os.environ["LD_LIBRARY_PATH"] = new_path
                # Also try adding to sys.path just in case
                # sys.path.extend(libs_to_add)
                print(f"Updated LD_LIBRARY_PATH with: {libs_to_add}")
                break
except Exception as e:
    print(f"Warning: Failed to auto-configure CUDA paths: {e}")

import onnxruntime
import onnx
import logging
import matplotlib
# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ultralytics import YOLO
from .vector_db import FaceDB

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self):
        # Locate models directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        
        self.yolo_path = os.path.join(models_dir, "yolov8n-face.pt")
        self.arcface_path = os.path.join(models_dir, "arcface.onnx")
        
        logger.info(f"Loading YOLO from {self.yolo_path}...")
        self.yolo = YOLO(self.yolo_path)
        
        # Force YOLO to use GPU if available
        import torch
        if torch.cuda.is_available():
            self.yolo.to('cuda')
            logger.info("✅ YOLO loaded with CUDA acceleration")
        else:
            logger.warning("⚠️ CUDA not available for YOLO, using CPU")
        
        logger.info(f"Loading ArcFace from {self.arcface_path}...")
        self.use_gpu = torch.cuda.is_available()
        
        if self.use_gpu:
            # Use PyTorch with GPU for ArcFace
            try:
                from onnx2torch import convert
                onnx_model = onnx.load(self.arcface_path)
                self.arcface_model = convert(onnx_model)
                self.arcface_model = self.arcface_model.to('cuda').eval()
                self.arcface_backend = 'torch'
                logger.info("✅ ArcFace using PyTorch GPU")
            except Exception as e:
                logger.warning(f"⚠️ PyTorch conversion failed: {e}, using ONNX CPU")
                self.ort_session = onnxruntime.InferenceSession(self.arcface_path, providers=['CPUExecutionProvider'])
                self.input_name = self.ort_session.get_inputs()[0].name
                self.arcface_backend = 'onnx'
        else:
            self.ort_session = onnxruntime.InferenceSession(self.arcface_path, providers=['CPUExecutionProvider'])
            self.input_name = self.ort_session.get_inputs()[0].name
            self.arcface_backend = 'onnx'
            logger.warning("⚠️ No GPU available, using CPU")
        
        self.face_db = FaceDB()
        
    def _process_frame_batch(self, frame_batch, detections, start_frame_idx):
        """Process batch of frames for GPU efficiency."""
        device = 'cuda' if hasattr(self.yolo.model, 'device') and 'cuda' in str(self.yolo.model.device) else 'cpu'
        results = self.yolo(frame_batch, verbose=False, device=device, max_det=20, conf=0.5, iou=0.7, stream=True, agnostic_nms=True)
        
        for frame_idx, (frame, result) in enumerate(zip(frame_batch, results)):
            actual_frame_idx = start_frame_idx + frame_idx
            
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0: continue
                
                # Store face crop for batch embedding generation later
                detections.append({
                    "frame_idx": actual_frame_idx,
                    "bbox": (x1, y1, x2, y2),
                    "face_crop": face_crop
                })

    def get_embedding_batch(self, face_crops):
        """Generate embeddings for batch of face crops on GPU."""
        try:
            if self.arcface_backend == 'torch' and self.use_gpu:
                # GPU batch processing with PyTorch
                batch = []
                for face_img in face_crops:
                    face_img = cv2.resize(face_img, (112, 112))
                    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    img = np.transpose(img, (2, 0, 1))
                    img = (img - 127.5) / 128.0
                    batch.append(img)
                
                batch_tensor = torch.from_numpy(np.array(batch, dtype=np.float32)).to('cuda')
                
                with torch.no_grad():
                    embeddings = self.arcface_model(batch_tensor)
                    embeddings = embeddings.cpu().numpy()
                
                # Normalize
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-10)
                return embeddings
            else:
                # CPU fallback
                embeddings = []
                for face_img in face_crops:
                    face_img = cv2.resize(face_img, (112, 112))
                    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    img = np.transpose(img, (2, 0, 1))
                    img = np.expand_dims(img, axis=0)
                    img = (img - 127.5) / 128.0
                    img = img.astype(np.float32)
                    
                    embedding = self.ort_session.run(None, {self.input_name: img})[0]
                    embedding = embedding.flatten()
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    embeddings.append(embedding)
                
                return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            return np.zeros((len(face_crops), 512))
    
    def get_embedding(self, face_img):
        """Generate single embedding (uses batch internally)."""
        return self.get_embedding_batch([face_img])[0]

    def enroll_face(self, image_path, person_id):
        """
        Detects face in image and adds to DB.
        Assumes single face in enrollment image.
        """
        logger.info(f"Starting enrollment for person_id: {person_id}")
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            raise ValueError("Could not read image")
            
        logger.info(f"Running face detection on enrollment image...")
        # Ensure YOLO uses GPU for enrollment too
        device = 'cuda' if hasattr(self.yolo.model, 'device') and 'cuda' in str(self.yolo.model.device) else 'cpu'
        results = self.yolo(img, device=device, verbose=False, max_det=5, conf=0.5, stream=True, agnostic_nms=True)
        
        # Take the detection with highest confidence or largest area
        best_face = None
        max_area = 0
        face_count = 0
        
        for result in results:
            for box in result.boxes:
                face_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_face = img[y1:y2, x1:x2]
        
        logger.info(f"Detected {face_count} faces in enrollment image")
        if best_face is None:
            logger.error("No face detected in enrollment image")
            raise ValueError("No face detected in enrollment image")
            
        logger.info(f"Generating embedding for {person_id}...")
        embedding = self.get_embedding(best_face)
        logger.info(f"Adding {person_id} to vector database...")
        self.face_db.add_person(person_id, embedding)
        logger.info(f"Successfully enrolled {person_id}")
        return True

    def generate_cluster_plot(self, embeddings, labels, output_path):
        """
        Generates a 2D PCA plot of the face clusters.
        """
        try:
            if len(embeddings) < 2:
                return None
                
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(embeddings)
            
            plt.figure(figsize=(10, 8))
            unique_labels = set(labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    c = 'k' # Black for noise
                    marker = 'x'
                    label_text = "Noise"
                else:
                    c = color
                    marker = 'o'
                    label_text = f"Cluster {label}"
                    
                indices = [i for i, l in enumerate(labels) if l == label]
                plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1], c=[c], label=label_text, marker=marker, alpha=0.6)
                
            plt.title('Face Clusters Visualization (PCA)')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(output_path)
            plt.close()
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate cluster plot: {e}")
            return None

    def process_video(self, video_path, output_path, progress_callback=None):
        def update_progress(percent, msg):
            if progress_callback:
                progress_callback(percent, msg)
            logger.info(f"{percent}% - {msg}")

        cap = cv2.VideoCapture(video_path)
        detections = [] # {frame_idx, bbox, embedding}
        frames_count = 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        update_progress(0, f"Starting processing for {total_frames} frames...")
        
        # 1. Detection & Embedding (GPU Batch Processing)
        batch_size = 1  # Process one frame at a time to prevent NMS timeout
        frame_batch = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Process remaining frames
                if frame_batch:
                    self._process_frame_batch(frame_batch, detections, frames_count - len(frame_batch))
                break
                
            frame_batch.append(frame)
            frames_count += 1
            
            # Process when batch is full
            if len(frame_batch) == batch_size:
                batch_start = frames_count - batch_size
                self._process_frame_batch(frame_batch, detections, batch_start)
                frame_batch = []
                
            if frames_count % 25 == 0:
                prog = int((frames_count / total_frames) * 50)
                logger.info(f"Processed frame {frames_count}/{total_frames} (Found {len(detections)} faces so far)")
                if progress_callback:
                     progress_callback(prog, f"Scanning frames: {frames_count}/{total_frames}")
                
        cap.release()
        
        if not detections:
            update_progress(100, "No faces found in video.")
            return False

        # 2. Batch Embedding Generation on GPU
        update_progress(52, f"Generating embeddings for {len(detections)} faces on GPU...")
        backend = "GPU (PyTorch)" if self.arcface_backend == 'torch' and self.use_gpu else "CPU (ONNX)"
        logger.info(f"Generating embeddings for {len(detections)} face crops using {backend}...")
        
        embeddings = []
        batch_size = 64 if self.arcface_backend == 'torch' and self.use_gpu else 1  # GPU can handle larger batches
        
        for i in range(0, len(detections), batch_size):
            batch_end = min(i + batch_size, len(detections))
            face_batch = [detections[j]["face_crop"] for j in range(i, batch_end)]
            
            batch_embeddings = self.get_embedding_batch(face_batch)
            embeddings.extend(batch_embeddings)
            
            for j, emb in enumerate(batch_embeddings):
                detections[i + j]["embedding"] = emb
            
            if (i + batch_size) % 100 == 0:
                prog = 52 + int((i / len(detections)) * 3)
                update_progress(prog, f"Generated {i+batch_size}/{len(detections)} embeddings...")
        
        embeddings_array = np.array(embeddings)
        logger.info(f"✅ Generated {len(embeddings)} embeddings using GPU")
        
        # 3. Clustering
        update_progress(55, f"Clustering {len(embeddings)} faces...")
        logger.info(f"Starting clustering on {len(embeddings)} face embeddings...")
        
        # HDBSCAN
        if len(embeddings_array) >= 3:
            # Skip PCA - use raw embeddings for better quality
            logger.info("Skipping PCA - using raw 512-dim embeddings for clustering...")
            reduced_embeddings = embeddings_array

            # Force GPU HDBSCAN
            update_progress(58, f"Running GPU HDBSCAN clustering on {len(reduced_embeddings)} points...")
            
            if GPU_HDBSCAN_AVAILABLE:
                logger.info("Running GPU-accelerated HDBSCAN (cuML)...")
                import cudf
                reduced_df = cudf.DataFrame(reduced_embeddings)
                clusterer = cumlHDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_epsilon=0.5)
                labels = clusterer.fit_predict(reduced_df).to_numpy()
                logger.info("✅ GPU HDBSCAN completed")
            else:
                logger.info(f"Running CPU HDBSCAN (cuML not installed) on {len(reduced_embeddings)} points...")
                import hdbscan
                clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_epsilon=0.5, core_dist_n_jobs=-1)
                labels = clusterer.fit_predict(reduced_embeddings)
                logger.info("✅ CPU HDBSCAN completed")
            
            update_progress(59, "Clustering analysis complete...")
        else:
            labels = list(range(len(embeddings_array))) # Too few faces, treat unique
        
        update_progress(60, f"Found {len(set(labels))} unique clusters.")

        # Skip visualization for speed
        generated_plot = None

        # 4. Identification
        update_progress(70, "Identifying clusters against database...")
        cluster_map = {} # label -> person_id
        unique_labels = set(labels)
        
        logger.info(f"Identifying {len(unique_labels)} clusters against enrolled database...")
        
        for label in unique_labels:
            if label == -1:
                logger.info(f"Skipping noise cluster (label -1)")
                continue # Noise
            
            # Get mean embedding for this cluster
            cluster_indices = [i for i, x in enumerate(labels) if x == label]
            cluster_embeddings = [embeddings[i] for i in cluster_indices]
            logger.info(f"Cluster {label}: {len(cluster_indices)} faces")
            
            mean_emb = np.mean(cluster_embeddings, axis=0)
            
            # Renormalize mean ?
            norm = np.linalg.norm(mean_emb)
            if norm > 0: mean_emb = mean_emb / norm
            
            logger.info(f"Searching database for cluster {label}...")
            person_id = self.face_db.search_person(mean_emb)
            if person_id:
                cluster_map[label] = person_id
                logger.info(f"✅ Cluster {label} identified as: {person_id}")
            else:
                cluster_map[label] = f"Unknown_{label}"
                logger.info(f"❌ Cluster {label} not recognized - marked as Unknown_{label}")
        
        logger.info(f"Identification complete. Recognized: {sum(1 for v in cluster_map.values() if not v.startswith('Unknown'))}/{len(cluster_map)} clusters")
                
        # 5. Annotation
        update_progress(80, "Annotating video...")
        logger.info(f"Starting video annotation and output generation...")
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Output video specs: {width}x{height} @ {fps:.1f}fps")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        current_frame = 0
        det_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Draw detections
            # detections list corresponds to our sequential processing
            # We need to map back which detection corresponds to which label
            
            # Optimization: since we appended sequentially, we can just iterate
            while det_idx < len(detections) and detections[det_idx]["frame_idx"] == current_frame:
                d = detections[det_idx]
                label = labels[det_idx]
                
                name = "Unknown"
                if label != -1:
                    name = cluster_map.get(label, "Unknown")
                
                x1, y1, x2, y2 = d["bbox"]
                
                # Color based on recognized or not
                color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                det_idx += 1
            
            out.write(frame)
            current_frame += 1
            
            if current_frame % 25 == 0:  # More frequent updates
                 prog = 80 + int((current_frame / total_frames) * 19)
                 if progress_callback:
                    progress_callback(prog, f"Annotating frame {current_frame}/{total_frames}")

        cap.release()
        out.release()
        
        update_progress(100, "Processing completed.")
        logger.info(f"Completed. Saved to {output_path}")
        
        return True, generated_plot
