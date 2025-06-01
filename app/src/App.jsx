import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Upload, Download, Settings, Zap, Square, Play, Pause, RotateCcw, FileImage, Sparkles, Target, TrendingUp } from 'lucide-react';

const TextDetectionApp = () => {
  const [mode, setMode] = useState('upload');
  const [isStreaming, setIsStreaming] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [detectedText, setDetectedText] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [confidence, setConfidence] = useState(0.7);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [imageNaturalSize, setImageNaturalSize] = useState({ width: 0, height: 0 });
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const streamRef = useRef(null);
  const imageRef = useRef(null);

  
  const detectText = useCallback(async (imageData) => {
    setIsProcessing(true);
  
    try {
      // Remove data URL prefix to get just the base64 string
      const base64String = imageData.replace(/^data:image\/[a-z]+;base64,/, '');
      
      const response = await fetch('http://0.0.0.0:8081/detect/base64?', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64String
        }),
      });
  
      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }
  
      const data = await response.json();
      console.log('Detection response:', data);
      
      // Process the detection results based on your API structure
      const processedDetections = [];
      
      if (data && data.detections) {
        data.detections.forEach((detection) => {
          if (detection.text && detection.confidence && detection.box) {
            // detection.box is an array of 4 points, each point has [x, y]
            // Extract all x and y coordinates from the 4 points
            const points = detection.box;
            const xCoords = points.map(point => point[0]);
            const yCoords = points.map(point => point[1]);
            
            // Find min/max to create bounding rectangle
            const minX = Math.min(...xCoords);
            const maxX = Math.max(...xCoords);
            const minY = Math.min(...yCoords);
            const maxY = Math.max(...yCoords);
            
            processedDetections.push({
              text: detection.text,
              confidence: detection.confidence,
              bbox: {
                x: minX,
                y: minY,
                width: maxX - minX,
                height: maxY - minY
              },
              // Keep original points for debugging if needed
              originalPoints: points
            });
          }
        });
      }
    
    console.log('Processed detections:', processedDetections);
    setDetectedText(processedDetections);
  } catch (error) {
    console.error('Error detecting text:', error);
    setDetectedText([]);
    // For demo purposes, let's add some mock data with percentage positioning
    const mockData = [
      {
        id: 0,
        text: "Sample Text",
        confidence: 0.95,
        bbox: { x: 100, y: 50, width: 200, height: 30 }
      },
      {
        id: 1,
        text: "Another Detection",
        confidence: 0.87,
        bbox: { x: 150, y: 120, width: 180, height: 25 }
      }
    ];
    // setDetectedText(mockData); // Uncomment for testing
  } finally {
    setIsProcessing(false);
  }
}, [confidence]);
  
  // Convert pixel coordinates to percentage based on image natural size
  const convertToPercentage = (bbox, imageWidth, imageHeight) => {
    return {
      x: (bbox.x / imageWidth) * 100,
      y: (bbox.y / imageHeight) * 100,
      width: (bbox.width / imageWidth) * 100,
      height: (bbox.height / imageHeight) * 100
    };
  };

  // Handle image load to get natural dimensions
  const handleImageLoad = (e) => {
    const { naturalWidth, naturalHeight } = e.target;
    setImageNaturalSize({ width: naturalWidth, height: naturalHeight });
    
    // Update existing detections to percentage if we have them
    if (detectedText.length > 0 && naturalWidth && naturalHeight) {
      const updatedDetections = detectedText.map(detection => ({
        ...detection,
        bboxPercent: convertToPercentage(detection.bbox, naturalWidth, naturalHeight)
      }));
      setDetectedText(updatedDetections);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'environment'
        } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsStreaming(true);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Unable to access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsStreaming(false);
  };

  const captureFrame = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const ctx = canvas.getContext('2d');
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      
      const imageData = canvas.toDataURL('image/jpeg');
      setCapturedImage(imageData);
      setImageNaturalSize({ width: video.videoWidth, height: video.videoHeight });
      detectText(imageData);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setCapturedImage(e.target.result);
        // Image dimensions will be set when the image loads
        detectText(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const downloadResults = () => {
    const results = {
      timestamp: new Date().toISOString(),
      detectedText: detectedText,
      imageNaturalSize: imageNaturalSize,
      settings: { confidence, showBoundingBoxes }
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `text-detection-results-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const resetDetection = () => {
    setCapturedImage(null);
    setDetectedText([]);
    setIsProcessing(false);
    setImageNaturalSize({ width: 0, height: 0 });
  };

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-950 relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-cyan-500/5 rounded-full blur-2xl animate-pulse delay-500"></div>
      </div>

      {/* Header */}
      <div className="relative z-10 backdrop-blur-xl bg-black/30 border-b border-white/10 shadow-2xl">
        <div className="max-w-8xl mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3 sm:space-x-4">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-purple-500 rounded-2xl blur opacity-75"></div>
                <div className="relative p-2 sm:p-3 bg-gradient-to-r from-cyan-500 to-purple-600 rounded-2xl">
                  <Sparkles className="w-5 h-5 sm:w-7 sm:h-7 text-white" />
                </div>
              </div>
              <div>
                <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold bg-gradient-to-r from-white via-cyan-100 to-purple-200 bg-clip-text text-transparent">
                  AI Vision Studio
                </h1>
                <p className="text-sm sm:text-base text-cyan-200/80 font-medium">Advanced text recognition powered by AI</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-2 sm:space-x-3">
              <div className="hidden md:flex items-center space-x-2 bg-white/5 backdrop-blur-sm rounded-xl px-3 sm:px-4 py-2 border border-white/10">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-green-300 text-xs sm:text-sm font-medium">Live Detection</span>
              </div>
              <button
                onClick={() => setMode(mode === 'upload' ? 'camera' : 'upload')}
                className="group relative p-2 sm:p-3 bg-white/10 hover:bg-white/20 rounded-xl transition-all duration-300 border border-white/20 hover:border-white/30"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <div className="relative">
                  {mode === 'upload' ? 
                    <Camera className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-300" /> : 
                    <Upload className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-300" />
                  }
                </div>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="relative z-10 max-w-8xl mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-8">
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-4 sm:gap-6 lg:gap-8">
          
          {/* Main Content */}
          <div className="xl:col-span-3 space-y-4 sm:space-y-6 lg:space-y-8">
            
            {/* Enhanced Mode Toggle */}
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-3xl blur-xl"></div>
              <div className="relative flex bg-white/10 backdrop-blur-xl rounded-2xl sm:rounded-3xl p-1.5 sm:p-2 border border-white/20 shadow-2xl">
                <button
                  onClick={() => setMode('upload')}
                  className={`flex-1 flex items-center justify-center space-x-2 sm:space-x-3 py-3 sm:py-4 px-4 sm:px-6 rounded-xl sm:rounded-2xl transition-all duration-300 group ${
                    mode === 'upload' 
                      ? 'bg-gradient-to-r from-cyan-500 to-purple-600 text-white shadow-2xl transform scale-[1.02]' 
                      : 'text-gray-300 hover:text-white hover:bg-white/10'
                  }`}
                >
                  <FileImage className={`w-4 h-4 sm:w-5 sm:h-5 ${mode === 'upload' ? 'drop-shadow-lg' : ''}`} />
                  <span className="text-sm sm:text-base font-semibold">Upload Image</span>
                  {mode === 'upload' && <div className="w-2 h-2 bg-white/50 rounded-full animate-pulse"></div>}
                </button>
                <button
                  onClick={() => setMode('camera')}
                  className={`flex-1 flex items-center justify-center space-x-2 sm:space-x-3 py-3 sm:py-4 px-4 sm:px-6 rounded-xl sm:rounded-2xl transition-all duration-300 group ${
                    mode === 'camera' 
                      ? 'bg-gradient-to-r from-cyan-500 to-purple-600 text-white shadow-2xl transform scale-[1.02]' 
                      : 'text-gray-300 hover:text-white hover:bg-white/10'
                  }`}
                >
                  <Camera className={`w-4 h-4 sm:w-5 sm:h-5 ${mode === 'camera' ? 'drop-shadow-lg' : ''}`} />
                  <span className="text-sm sm:text-base font-semibold">Live Camera</span>
                  {mode === 'camera' && <div className="w-2 h-2 bg-white/50 rounded-full animate-pulse"></div>}
                </button>
              </div>
            </div>

            {/* Enhanced Camera/Upload Section */}
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-white/10 rounded-2xl sm:rounded-3xl blur-xl"></div>
              <div className="relative bg-white/10 backdrop-blur-xl rounded-2xl sm:rounded-3xl p-4 sm:p-6 lg:p-8 border border-white/20 shadow-2xl">
                {mode === 'camera' ? (
                  <div className="space-y-4 sm:space-y-6">
                    <div className="relative aspect-video bg-black/50 rounded-xl sm:rounded-2xl overflow-hidden border border-white/20 shadow-inner">
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="w-full h-full object-cover"
                      />
                      {!isStreaming && (
                        <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-slate-900/90 to-indigo-900/90">
                          <div className="text-center space-y-4 sm:space-y-6 p-4">
                            <div className="relative">
                              <div className="absolute inset-0 bg-cyan-400/20 rounded-full blur-2xl"></div>
                              <Camera className="relative w-12 h-12 sm:w-16 lg:w-20 sm:h-16 lg:h-20 text-cyan-300 mx-auto drop-shadow-2xl" />
                            </div>
                            <div className="space-y-2">
                              <p className="text-white text-base sm:text-lg font-semibold">Camera Ready</p>
                              <p className="text-cyan-200/80 text-sm sm:text-base">Start capturing and analyzing text in real-time</p>
                            </div>
                            <button
                              onClick={startCamera}
                              className="group relative bg-gradient-to-r from-cyan-500 to-purple-600 text-white px-6 sm:px-8 py-3 sm:py-4 rounded-xl sm:rounded-2xl hover:shadow-2xl transition-all duration-300 font-semibold text-sm sm:text-base"
                            >
                              <div className="absolute inset-0 bg-white/20 rounded-xl sm:rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
                              <div className="relative flex items-center space-x-2">
                                <Play className="w-4 h-4 sm:w-5 sm:h-5" />
                                <span>Start Camera</span>
                              </div>
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                    
                    {isStreaming && (
                      <div className="flex flex-col sm:flex-row justify-center space-y-3 sm:space-y-0 sm:space-x-4">
                        <button
                          onClick={captureFrame}
                          disabled={isProcessing}
                          className="group relative bg-gradient-to-r from-emerald-500 to-cyan-500 text-white px-6 sm:px-8 py-3 rounded-xl sm:rounded-2xl hover:shadow-2xl transition-all duration-300 disabled:opacity-50 font-semibold text-sm sm:text-base"
                        >
                          <div className="absolute inset-0 bg-white/20 rounded-xl sm:rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
                          <div className="relative flex items-center justify-center space-x-2">
                            <Target className="w-4 h-4 sm:w-5 sm:h-5" />
                            <span>Capture & Analyze</span>
                          </div>
                        </button>
                        <button
                          onClick={stopCamera}
                          className="group relative bg-gradient-to-r from-red-500 to-pink-500 text-white px-6 sm:px-8 py-3 rounded-xl sm:rounded-2xl hover:shadow-2xl transition-all duration-300 font-semibold text-sm sm:text-base"
                        >
                          <div className="absolute inset-0 bg-white/20 rounded-xl sm:rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
                          <div className="relative flex items-center justify-center space-x-2">
                            <Pause className="w-4 h-4 sm:w-5 sm:h-5" />
                            <span>Stop Camera</span>
                          </div>
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-4 sm:space-y-6">
                    <div
                      onClick={() => fileInputRef.current?.click()}
                      className="group relative aspect-video border-2 border-dashed border-cyan-400/50 hover:border-cyan-400 rounded-xl sm:rounded-2xl cursor-pointer transition-all duration-300 bg-gradient-to-br from-black/20 to-indigo-900/20 hover:from-cyan-500/10 hover:to-purple-500/10"
                    >
                      <div className="absolute inset-0 flex items-center justify-center p-4">
                        <div className="text-center space-y-4 sm:space-y-6">
                          <div className="relative">
                            <div className="absolute inset-0 bg-cyan-400/20 rounded-full blur-2xl group-hover:blur-3xl transition-all"></div>
                            <Upload className="relative w-12 h-12 sm:w-16 lg:w-20 sm:h-16 lg:h-20 text-cyan-300 mx-auto drop-shadow-2xl group-hover:scale-110 transition-transform" />
                          </div>
                          <div className="space-y-2">
                            <p className="text-white text-base sm:text-lg font-semibold">Drop your image here</p>
                            <p className="text-cyan-200/80 text-sm sm:text-base">or click to browse files</p>
                            <p className="text-xs sm:text-sm text-gray-400">PNG, JPG, GIF up to 10MB</p>
                          </div>
                        </div>
                      </div>
                    </div>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                  </div>
                )}
              </div>
            </div>

            {/* Enhanced Results Section */}
            {capturedImage && (
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-white/10 rounded-2xl sm:rounded-3xl blur-xl"></div>
                <div className="relative bg-white/10 backdrop-blur-xl rounded-2xl sm:rounded-3xl p-4 sm:p-6 lg:p-8 border border-white/20 shadow-2xl">
                  <div className="flex items-center justify-between mb-4 sm:mb-6">
                    <div className="flex items-center space-x-3">
                      <div className="p-2 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-xl">
                        <Target className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
                      </div>
                      <h3 className="text-lg sm:text-xl font-bold text-white">Detection Results</h3>
                    </div>
                    <div className="flex space-x-2 sm:space-x-3">
                      <button
                        onClick={downloadResults}
                        disabled={detectedText.length === 0}
                        className="group relative p-2 sm:p-3 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 rounded-xl transition-all duration-300 disabled:opacity-50 border border-emerald-500/30"
                      >
                        <Download className="w-4 h-4 sm:w-5 sm:h-5" />
                      </button>
                      <button
                        onClick={resetDetection}
                        className="group relative p-2 sm:p-3 bg-gray-500/20 hover:bg-gray-500/30 text-gray-300 rounded-xl transition-all duration-300 border border-gray-500/30"
                      >
                        <RotateCcw className="w-4 h-4 sm:w-5 sm:h-5" />
                      </button>
                    </div>
                  </div>
                  
                  <div className="relative">
                    <img
                      ref={imageRef}
                      src={capturedImage}
                      alt="Captured"
                      className="w-full h-auto rounded-xl sm:rounded-2xl shadow-2xl border border-white/20"
                      onLoad={handleImageLoad}
                    />
                    
                    {isProcessing && (
                      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center rounded-xl sm:rounded-2xl">
                        <div className="text-center space-y-4 p-4">
                          <div className="relative">
                            <div className="animate-spin rounded-full h-12 w-12 sm:h-16 sm:w-16 border-b-4 border-cyan-400 mx-auto"></div>
                            <div className="absolute inset-0 animate-ping rounded-full h-12 w-12 sm:h-16 sm:w-16 border-4 border-cyan-400/20"></div>
                          </div>
                          <div className="space-y-2">
                            <p className="text-white font-semibold text-base sm:text-lg">AI Processing</p>
                            <p className="text-cyan-200 text-sm sm:text-base">Analyzing image for text...</p>
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {showBoundingBoxes && detectedText.map((text) => {
                      // Use percentage positioning if available, otherwise convert from pixel
                      const bboxToUse = text.bboxPercent || 
                        (imageNaturalSize.width && imageNaturalSize.height ? 
                          convertToPercentage(text.bbox, imageNaturalSize.width, imageNaturalSize.height) : 
                          null);
                      
                      if (!bboxToUse) return null;
                      
                      return (
                        <div
                          className="absolute border-2 border-cyan-400 bg-cyan-400/10 backdrop-blur-sm rounded-lg"
                          style={{
                            left: `${bboxToUse.x}%`,
                            top: `${bboxToUse.y}%`,
                            width: `${bboxToUse.width}%`,
                            height: `${bboxToUse.height}%`,
                          }}
                        >
                          <div className="absolute -top-6 sm:-top-8 left-0 bg-gradient-to-r from-cyan-500 to-purple-600 text-white text-xs px-2 sm:px-3 py-1 rounded-lg whitespace-nowrap font-semibold shadow-2xl max-w-xs truncate">
                            {text.text} ({Math.round(text.confidence * 100)}%)
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Enhanced Sidebar */}
          <div className="space-y-4 sm:space-y-6 lg:space-y-8">
            
            {/* Enhanced Settings */}
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-white/10 rounded-2xl sm:rounded-3xl blur-xl"></div>
              <div className="relative bg-white/10 backdrop-blur-xl rounded-2xl sm:rounded-3xl p-4 sm:p-6 border border-white/20 shadow-2xl">
                <div className="flex items-center space-x-3 mb-4 sm:mb-6">
                  <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl">
                    <Settings className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
                  </div>
                  <h3 className="text-base sm:text-lg font-bold text-white">Settings</h3>
                </div>
                
                <div className="space-y-4 sm:space-y-6">
                  <div>
                    <label className="block text-sm font-semibold text-cyan-200 mb-3">
                      Confidence Threshold
                    </label>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-gray-400">Low</span>
                        <span className="text-base sm:text-lg font-bold text-white bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                          {Math.round(confidence * 100)}%
                        </span>
                        <span className="text-xs text-gray-400">High</span>
                      </div>
                      <div className="relative">
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={confidence}
                          onChange={(e) => setConfidence(parseFloat(e.target.value))}
                          className="w-full h-3 bg-gradient-to-r from-gray-700 to-gray-600 rounded-lg appearance-none cursor-pointer slider"
                        />
                        <div 
                          className="absolute top-0 left-0 h-3 bg-gradient-to-r from-cyan-500 to-purple-600 rounded-lg pointer-events-none"
                          style={{ width: `${confidence * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between p-3 sm:p-4 bg-white/5 rounded-xl sm:rounded-2xl border border-white/10">
                    <div className="flex items-center space-x-3">
                      <div className="p-1.5 sm:p-2 bg-cyan-500/20 rounded-lg">
                        <Target className="w-3 h-3 sm:w-4 sm:h-4 text-cyan-300" />
                      </div>
                      <span className="text-white font-medium text-sm sm:text-base">Bounding Boxes</span>
                    </div>
                    <button
                      onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
                      className={`relative inline-flex h-6 w-10 sm:h-7 sm:w-12 items-center rounded-full transition-all duration-300 ${
                        showBoundingBoxes 
                          ? 'bg-gradient-to-r from-cyan-500 to-purple-600 shadow-lg' 
                          : 'bg-gray-600'
                      }`}
                    >
                      <span
                        className={`inline-block h-4 w-4 sm:h-5 sm:w-5 transform rounded-full bg-white transition-transform duration-300 shadow-lg ${
                          showBoundingBoxes ? 'translate-x-5 sm:translate-x-6' : 'translate-x-1'
                        }`}
                      />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Enhanced Text List */}
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-white/10 rounded-2xl sm:rounded-3xl blur-xl"></div>
              <div className="relative bg-white/10 backdrop-blur-xl rounded-2xl sm:rounded-3xl p-4 sm:p-6 border border-white/20 shadow-2xl">
                <div className="flex items-center space-x-3 mb-4 sm:mb-6">
                  <div className="p-2 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-xl">
                    <FileImage className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
                  </div>
                  <h3 className="text-base sm:text-lg font-bold text-white">Detected Text</h3>
                </div>
                
                {detectedText.length === 0 ? (
                  <div className="text-center py-8 sm:py-12 space-y-4">
                    <div className="w-12 h-12 sm:w-16 sm:h-16 bg-gray-500/20 rounded-full flex items-center justify-center mx-auto">
                      <FileImage className="w-6 h-6 sm:w-8 sm:h-8 text-gray-400" />
                    </div>
                    <p className="text-gray-400 text-sm sm:text-base">No text detected yet</p>
                  </div>
                ) : (
                  <div className="space-y-3 max-h-64 sm:max-h-96 overflow-y-auto">
                    {detectedText.map((text) => (
                      <div
                        className="group relative bg-white/5 hover:bg-white/10 rounded-xl sm:rounded-2xl p-3 sm:p-4 border border-white/10 hover:border-white/20 transition-all duration-300"
                      >
                        <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 to-purple-500/5 rounded-xl sm:rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
                        <div className="relative">
                          <div className="flex items-start justify-between mb-2">
                            <span className="text-white font-semibold flex-1 pr-2 text-sm sm:text-base break-words">{text.text}</span>
                            <div className="flex items-center space-x-2">
                              <div className={`px-2 py-1 rounded-lg text-xs font-bold ${
                                text.confidence >= 0.9 
                                  ? 'bg-emerald-500/20 text-emerald-300' 
                                  : text.confidence >= 0.7 
                                  ? 'bg-yellow-500/20 text-yellow-300' 
                                  : 'bg-red-500/20 text-red-300'
                              }`}>
                                {Math.round(text.confidence * 100)}%
                              </div>
                            </div>
                          </div>
                          <div className="text-xs text-gray-400 bg-black/20 rounded-lg px-2 py-1">
                            {text.bboxPercent 
                              ? `Position: (${text.bboxPercent.x.toFixed(1)}%, ${text.bboxPercent.y.toFixed(1)}%) • Size: ${text.bboxPercent.width.toFixed(1)}%×${text.bboxPercent.height.toFixed(1)}%`
                              : `Position: (${text.bbox.x}, ${text.bbox.y}) • Size: ${text.bbox.width}×${text.bbox.height}`
                            }
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Enhanced Stats */}
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-white/10 rounded-2xl sm:rounded-3xl blur-xl"></div>
              <div className="relative bg-white/10 backdrop-blur-xl rounded-2xl sm:rounded-3xl p-4 sm:p-6 border border-white/20 shadow-2xl">
                <div className="flex items-center space-x-3 mb-4 sm:mb-6">
                  <div className="p-2 bg-gradient-to-r from-pink-500 to-purple-500 rounded-xl">
                    <TrendingUp className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
                  </div>
                  <h3 className="text-base sm:text-lg font-bold text-white">Analytics</h3>
                </div>
                <div className="space-y-3 sm:space-y-4">
                  <div className="flex justify-between items-center p-3 bg-white/5 rounded-xl border border-white/10">
                    <span className="text-cyan-200 font-medium text-sm sm:text-base">Text Found</span>
                    <span className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                      {detectedText.length}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-white/5 rounded-xl border border-white/10">
                    <span className="text-cyan-200 font-medium text-sm sm:text-base">Avg Confidence</span>
                    <span className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                      {detectedText.length > 0 
                        ? Math.round((detectedText.reduce((sum, text) => sum + text.confidence, 0) / detectedText.length) * 100) + '%'
                        : '0%'
                      }
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-white/5 rounded-xl border border-white/10">
                    <span className="text-cyan-200 font-medium text-sm sm:text-base">Status</span>
                    <div className="flex items-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-yellow-400 animate-pulse' : 'bg-emerald-400'}`}></div>
                      <span className={`font-bold text-sm sm:text-base ${isProcessing ? 'text-yellow-400' : 'text-emerald-400'}`}>
                        {isProcessing ? 'Processing' : 'Ready'}
                      </span>
                    </div>
                  </div>
                  {imageNaturalSize.width > 0 && (
                    <div className="flex justify-between items-center p-3 bg-white/5 rounded-xl border border-white/10">
                      <span className="text-cyan-200 font-medium text-sm sm:text-base">Image Size</span>
                      <span className="text-sm sm:text-base font-bold text-white">
                        {imageNaturalSize.width}×{imageNaturalSize.height}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
};

export default TextDetectionApp;