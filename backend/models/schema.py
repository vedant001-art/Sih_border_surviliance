from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, JSON, Text, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from backend.core.database import Base

class UserRole(str, enum.Enum):
    ADMIN = "ADMIN"
    SUPERVISOR = "SUPERVISOR"
    OPERATOR = "OPERATOR"
    ANALYST = "ANALYST"

class AlertSeverity(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertStatus(str, enum.Enum):
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED = "RESOLVED"

class TrackStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    LOST = "LOST"
    ENDED = "ENDED"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(Enum(UserRole), default=UserRole.OPERATOR)
    is_active = Column(Boolean, default=True)

class Camera(Base):
    __tablename__ = "cameras"
    id = Column(String, primary_key=True, index=True) # e.g. CAM-01
    camera_code = Column(String, nullable=True)
    name = Column(String)
    source_type = Column(String, nullable=True)
    source_url = Column(String, nullable=True)
    location_name = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    status = Column(String, default="ONLINE")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    is_active = Column(Boolean, default=True)
    
    tracks = relationship("Track", back_populates="camera")
    events = relationship("Event", back_populates="camera")
    vehicles = relationship("Vehicle", back_populates="camera")

class Track(Base):
    __tablename__ = "tracks"
    id = Column(Integer, primary_key=True, index=True)
    local_track_id = Column(Integer, index=True)
    camera_id = Column(String, ForeignKey("cameras.id"), index=True)
    object_type = Column(String)
    class_name = Column(String)
    first_seen = Column(DateTime, default=datetime.now, index=True)
    last_seen = Column(DateTime, default=datetime.now, index=True)
    duration_seconds = Column(Float, default=0.0)
    initial_confidence = Column(Float)
    last_confidence = Column(Float)
    status = Column(Enum(TrackStatus), default=TrackStatus.ACTIVE)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    camera = relationship("Camera", back_populates="tracks")
    positions = relationship("TrackPosition", back_populates="track")
    vehicle = relationship("Vehicle", back_populates="track", uselist=False)

class TrackPosition(Base):
    __tablename__ = "track_positions"
    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, ForeignKey("tracks.id"), index=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    frame_number = Column(Integer, nullable=True)
    center_x = Column(Float)
    center_y = Column(Float)
    x1 = Column(Float)
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)
    confidence = Column(Float)
    
    track = relationship("Track", back_populates="positions")

class Vehicle(Base):
    __tablename__ = "vehicles"
    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, ForeignKey("tracks.id"))
    camera_id = Column(String, ForeignKey("cameras.id"), index=True)
    vehicle_type = Column(String)
    first_seen = Column(DateTime, default=datetime.now)
    last_seen = Column(DateTime, default=datetime.now)
    plate_number = Column(String, index=True, nullable=True)
    plate_confidence = Column(Float, nullable=True)
    status = Column(String, default="ACTIVE")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    track = relationship("Track", back_populates="vehicle")
    camera = relationship("Camera", back_populates="vehicles")
    anpr_records = relationship("ANPRRecord", back_populates="vehicle")

class ANPRRecord(Base):
    __tablename__ = "anpr_records"
    id = Column(Integer, primary_key=True, index=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), index=True, nullable=True)
    track_id = Column(Integer, ForeignKey("tracks.id"), index=True, nullable=True)
    camera_id = Column(String, ForeignKey("cameras.id"))
    plate_text = Column(String, index=True)
    raw_ocr_text = Column(String, nullable=True)
    ocr_confidence = Column(Float)
    plate_detection_confidence = Column(Float, nullable=True)
    first_seen = Column(DateTime, default=datetime.now)
    last_seen = Column(DateTime, default=datetime.now)
    observation_count = Column(Integer, default=1)
    plate_image_path = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    created_at = Column(DateTime, default=datetime.now)
    
    vehicle = relationship("Vehicle", back_populates="anpr_records")
    track = relationship("Track")

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, index=True) 
    camera_id = Column(String, ForeignKey("cameras.id"), index=True)
    track_id = Column(Integer, ForeignKey("tracks.id"), nullable=True, index=True)
    object_type = Column(String, nullable=True)
    zone_id = Column(String, nullable=True)
    severity = Column(Enum(AlertSeverity), default=AlertSeverity.LOW)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    title = Column(String)
    description = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    evidence_image_path = Column(String, nullable=True)
    evidence_video_path = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    summary_source = Column(String, nullable=True)
    status = Column(String, default="NEW")
    created_at = Column(DateTime, default=datetime.now)
    
    camera = relationship("Camera", back_populates="events")
    alert = relationship("Alert", back_populates="event", uselist=False)

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, ForeignKey("events.id"))
    severity = Column(Enum(AlertSeverity))
    title = Column(String, nullable=True)
    message = Column(String)
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=True)
    track_id = Column(Integer, ForeignKey("tracks.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.now)
    status = Column(Enum(AlertStatus), default=AlertStatus.NEW)
    acknowledged_by = Column(String, nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    event = relationship("Event", back_populates="alert")

class SystemLog(Base):
    __tablename__ = "system_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    level = Column(String)
    module = Column(String)
    message = Column(Text)
