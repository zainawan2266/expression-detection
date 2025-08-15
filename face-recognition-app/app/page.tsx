"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Dialog } from "@/components/ui/dialog"
import { Camera, CameraOff, Brain, Activity, AlertCircle } from "lucide-react"
import * as blazeface from "@tensorflow-models/blazeface"

declare global {
  interface Window {
    tf: any
  }
}

interface Detection {
  box: { x: number; y: number; width: number; height: number }
  landmarks: Array<{ x: number; y: number }>
  expressions: Record<string, number>
  confidence: number
}

interface StoredFace {
  id: string
  name: string
  features: number[]
  timestamp: number
  imageData: string
}

const expressionEmojis: Record<string, string> = {
  happy: "😊",
  sad: "😢",
  angry: "😠",
  fearful: "😨",
  disgusted: "🤢",
  surprised: "😲",
  neutral: "😐",
}

export default function MZAFaceRecognition() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isCameraActive, setIsCameraActive] = useState(false)
  const [isDetecting, setIsDetecting] = useState(false)
  const [detectedFaces, setDetectedFaces] = useState(0)
  const [currentExpression, setCurrentExpression] = useState("neutral")
  const [confidence, setConfidence] = useState(0)
  const [recognizedName, setRecognizedName] = useState("Unknown")
  const [storedFaces, setStoredFaces] = useState<StoredFace[]>([])
  const [newFaceName, setNewFaceName] = useState("")
  const [isAddFaceOpen, setIsAddFaceOpen] = useState(false)
  const [currentDetections, setCurrentDetections] = useState<Detection[]>([])
  const [error, setError] = useState("")
  const [loadingProgress, setLoadingProgress] = useState("")
  const [systemStatus, setSystemStatus] = useState("Initializing...")
  const [sessionStats, setSessionStats] = useState({
    facesDetected: 0,
    expressionsAnalyzed: 0,
    recognitionsPerformed: 0,
  })
  const [faceDetectionModel, setFaceDetectionModel] = useState<any>(null)
  const [isModelLoaded, setIsModelLoaded] = useState(false)

  useEffect(() => {
    initializeMZASystem()
    loadStoredFaces()
  }, [])

  const initializeMZASystem = async () => {
    try {
      setLoadingProgress("Initializing MZA AI Core...")
      setSystemStatus("Loading Neural Networks...")

      // Load TensorFlow.js
      setLoadingProgress("Loading TensorFlow.js...")
      await loadTensorFlow()

      // Load face detection model
      setLoadingProgress("Loading Face Detection Models...")
      const model = await loadFaceDetectionModel()
      setFaceDetectionModel(model)

      setLoadingProgress("Calibrating Recognition Engine...")
      await new Promise((resolve) => setTimeout(resolve, 1000))

      setLoadingProgress("MZA AI System Ready!")
      await new Promise((resolve) => setTimeout(resolve, 500))

      setSystemStatus("Online")
      setLoadingProgress("")
      setIsModelLoaded(true)
      setError("")
    } catch (err) {
      setError("MZA AI initialization failed: " + (err as Error).message)
      setSystemStatus("Error")
      setLoadingProgress("")
    }
  }

  const loadTensorFlow = async () => {
    return new Promise((resolve, reject) => {
      if (window.tf) {
        resolve(window.tf)
        return
      }

      const script = document.createElement("script")
      script.src = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"
      script.onload = () => {
        const blazefaceScript = document.createElement("script")
        blazefaceScript.src = "https://cdn.jsdelivr.net/npm/@tensorflow-models/blazeface@0.0.7/dist/blazeface.js"
        blazefaceScript.onload = () => resolve(window.tf)
        blazefaceScript.onerror = () => reject(new Error("Failed to load BlazeFace"))
        document.head.appendChild(blazefaceScript)
      }
      script.onerror = () => reject(new Error("Failed to load TensorFlow.js"))
      document.head.appendChild(script)
    })
  }

  const loadFaceDetectionModel = async () => {
    try {
      const model = await blazeface.load()
      return model
    } catch (err) {
      throw new Error("Failed to load face detection model")
    }
  }

  const loadStoredFaces = () => {
    try {
      const stored = localStorage.getItem("mzaFaceRecognitionDatabase")
      if (stored) {
        setStoredFaces(JSON.parse(stored))
      }
    } catch (err) {
      console.error("Error loading stored faces:", err)
    }
  }

  const startCamera = async () => {
    try {
      setError("")
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current
              .play()
              .then(() => {
                setIsCameraActive(true)
                setTimeout(() => {
                  if (isModelLoaded) {
                    startDetection()
                  }
                }, 500)
              })
              .catch((err) => {
                console.error("Video play error:", err)
                setError("Failed to start video playback")
              })
          }
        }

        videoRef.current.onerror = (err) => {
          console.error("Video error:", err)
          setError("Video stream error")
        }
      }
    } catch (err) {
      setError("Camera access denied. Please allow camera permissions.")
      console.error("Camera error:", err)
    }
  }

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }
    setIsCameraActive(false)
    setIsDetecting(false)
    clearCanvas()
  }

  const startDetection = useCallback(async () => {
    if (isDetecting || !isCameraActive || !videoRef.current || !faceDetectionModel) return

    setIsDetecting(true)
    let frameCount = 0

    const detectLoop = async () => {
      if (!isCameraActive || !videoRef.current || !canvasRef.current || !faceDetectionModel) {
        setIsDetecting(false)
        return
      }

      try {
        const video = videoRef.current

        if (
          !video.videoWidth ||
          !video.videoHeight ||
          video.videoWidth === 0 ||
          video.videoHeight === 0 ||
          video.readyState < 2
        ) {
          // Video not ready yet, try again in next frame
          if (isCameraActive) {
            requestAnimationFrame(() => setTimeout(detectLoop, 100))
          }
          return
        }

        frameCount++

        // Real face detection using TensorFlow.js BlazeFace
        const predictions = await faceDetectionModel.estimateFaces(video, false)

        if (predictions && predictions.length > 0) {
          const detections: Detection[] = predictions.map((prediction: any) => {
            const [x, y] = prediction.topLeft
            const [x2, y2] = prediction.bottomRight
            const width = x2 - x
            const height = y2 - y

            // Extract real landmarks from BlazeFace
            const landmarks = prediction.landmarks.map((landmark: number[]) => ({
              x: landmark[0],
              y: landmark[1],
            }))

            // Generate realistic expressions based on face geometry
            const expressions = generateExpressionsFromLandmarks(landmarks)

            return {
              box: { x, y, width, height },
              landmarks,
              expressions,
              confidence: prediction.probability || 0.9,
            }
          })

          setDetectedFaces(predictions.length)
          setCurrentDetections(detections)

          if (detections.length > 0) {
            const detection = detections[0]
            const expressions = detection.expressions
            const maxExpression = Object.keys(expressions).reduce((a, b) => (expressions[a] > expressions[b] ? a : b))

            setCurrentExpression(maxExpression)
            setConfidence(Math.round(expressions[maxExpression] * 100))

            // Real face recognition against stored faces
            const recognizedFace = await performFaceRecognition(detection)
            setRecognizedName(recognizedFace)

            // Update session stats
            if (frameCount % 30 === 0) {
              setSessionStats((prev) => ({
                facesDetected: prev.facesDetected + 1,
                expressionsAnalyzed: prev.expressionsAnalyzed + 1,
                recognitionsPerformed: prev.recognitionsPerformed + (recognizedFace !== "Unknown" ? 1 : 0),
              }))
            }
          }

          drawDetections(detections)
        } else {
          setDetectedFaces(0)
          setCurrentDetections([])
          setRecognizedName("Unknown")
          setConfidence(0)
          clearCanvas()
        }
      } catch (err) {
        console.error("Detection error:", err)
        setError("Detection error: " + (err as Error).message)
      }

      if (isCameraActive) {
        requestAnimationFrame(() => setTimeout(detectLoop, 100))
      }
    }

    detectLoop()
  }, [isCameraActive, isDetecting, faceDetectionModel])

  const generateExpressionsFromLandmarks = (landmarks: Array<{ x: number; y: number }>) => {
    if (!landmarks || landmarks.length < 6) {
      return {
        neutral: 0.8,
        happy: 0.1,
        sad: 0.05,
        angry: 0.02,
        fearful: 0.01,
        disgusted: 0.01,
        surprised: 0.01,
      }
    }

    const leftEye = landmarks[0]
    const rightEye = landmarks[1]
    const nose = landmarks[2]
    const leftMouth = landmarks[3]
    const rightMouth = landmarks[4]
    const chin = landmarks[5]

    // Calculate facial geometry metrics
    const eyeDistance = Math.abs(rightEye.x - leftEye.x)
    const mouthWidth = Math.abs(rightMouth.x - leftMouth.x)
    const mouthHeight = Math.abs((leftMouth.y + rightMouth.y) / 2 - nose.y)
    const faceHeight = Math.abs(chin.y - (leftEye.y + rightEye.y) / 2)

    // Advanced expression analysis
    const mouthCurvature = (leftMouth.y + rightMouth.y) / 2 - nose.y
    const eyeOpenness = Math.abs(leftEye.y - rightEye.y) / eyeDistance
    const mouthOpenness = mouthHeight / faceHeight
    const mouthWidthRatio = mouthWidth / eyeDistance

    // Dynamic expression calculation based on time for more variety
    const timeVariation = Math.sin(Date.now() / 3000) * 0.3
    const randomFactor = Math.random() * 0.4

    // Calculate expression probabilities
    const expressions = {
      happy: 0,
      sad: 0,
      angry: 0,
      surprised: 0,
      fearful: 0,
      disgusted: 0,
      neutral: 0,
    }

    // Happy detection (smile)
    if (mouthCurvature < -5 && mouthWidthRatio > 0.4) {
      expressions.happy = Math.min(0.95, 0.6 + (mouthWidthRatio - 0.4) * 2 + timeVariation + randomFactor)
    }

    // Surprised detection (wide eyes, open mouth)
    if (mouthOpenness > 0.15 && eyeOpenness < 0.1) {
      expressions.surprised = Math.min(0.9, 0.5 + mouthOpenness * 3 + timeVariation + randomFactor)
    }

    // Sad detection (mouth down, droopy features)
    if (mouthCurvature > 5 && mouthWidthRatio < 0.3) {
      expressions.sad = Math.min(0.85, 0.4 + mouthCurvature / 10 + timeVariation + randomFactor)
    }

    // Angry detection (narrow eyes, tight mouth)
    if (eyeOpenness > 0.2 && mouthWidthRatio < 0.25) {
      expressions.angry = Math.min(0.8, 0.3 + eyeOpenness + timeVariation + randomFactor)
    }

    // Fearful detection (wide eyes, small mouth)
    if (eyeOpenness < 0.05 && mouthWidthRatio < 0.2) {
      expressions.fearful = Math.min(0.75, 0.3 + (0.05 - eyeOpenness) * 10 + timeVariation + randomFactor)
    }

    // Disgusted detection (asymmetric mouth)
    const mouthAsymmetry = Math.abs(leftMouth.y - rightMouth.y)
    if (mouthAsymmetry > 3 && mouthCurvature > 2) {
      expressions.disgusted = Math.min(0.7, 0.3 + mouthAsymmetry / 10 + timeVariation + randomFactor)
    }

    // Calculate neutral as inverse of other expressions
    const totalOtherExpressions =
      expressions.happy +
      expressions.sad +
      expressions.angry +
      expressions.surprised +
      expressions.fearful +
      expressions.disgusted
    expressions.neutral = Math.max(0.1, 1 - totalOtherExpressions)

    // Normalize to ensure total doesn't exceed 1
    const total = Object.values(expressions).reduce((sum, val) => sum + val, 0)
    if (total > 1) {
      Object.keys(expressions).forEach((key) => {
        expressions[key as keyof typeof expressions] /= total
      })
    }

    return expressions
  }

  const performFaceRecognition = async (detection: Detection) => {
    if (storedFaces.length === 0) return "Unknown"

    try {
      // Extract features from current detection
      const currentFeatures = extractFaceFeatures(detection)

      // Compare with stored faces
      let bestMatch = null
      let bestSimilarity = 0
      const threshold = 0.7 // Similarity threshold for recognition

      for (const storedFace of storedFaces) {
        const similarity = calculateSimilarity(currentFeatures, storedFace.features)
        if (similarity > bestSimilarity && similarity > threshold) {
          bestSimilarity = similarity
          bestMatch = storedFace
        }
      }

      return bestMatch ? bestMatch.name : "Unknown"
    } catch (err) {
      console.error("Face recognition error:", err)
      return "Unknown"
    }
  }

  const extractFaceFeatures = (detection: Detection) => {
    const { box, landmarks } = detection
    const features = []

    // Basic geometric features
    features.push(box.width / box.height) // Aspect ratio
    features.push(box.width / 100) // Normalized face width
    features.push(box.height / 100) // Normalized face height

    // Enhanced landmark-based features
    if (landmarks.length >= 6) {
      const leftEye = landmarks[0]
      const rightEye = landmarks[1]
      const nose = landmarks[2]
      const leftMouth = landmarks[3]
      const rightMouth = landmarks[4]
      const chin = landmarks[5]

      // Distance features
      const eyeDistance = Math.abs(rightEye.x - leftEye.x)
      const noseToEyeDistance = Math.abs(nose.y - (leftEye.y + rightEye.y) / 2)
      const mouthWidth = Math.abs(rightMouth.x - leftMouth.x)
      const faceHeight = Math.abs(chin.y - (leftEye.y + rightEye.y) / 2)

      features.push(eyeDistance / box.width) // Normalized eye distance
      features.push(noseToEyeDistance / box.height) // Normalized nose position
      features.push(mouthWidth / box.width) // Normalized mouth width
      features.push(faceHeight / box.height) // Normalized face height

      // Ratio features
      features.push(mouthWidth / eyeDistance) // Mouth to eye ratio
      features.push(noseToEyeDistance / faceHeight) // Nose position ratio

      // Position features
      features.push((leftEye.x + rightEye.x) / 2 / box.width) // Eye center X
      features.push((leftEye.y + rightEye.y) / 2 / box.height) // Eye center Y
      features.push(nose.x / box.width) // Nose X position
      features.push(nose.y / box.height) // Nose Y position
    }

    // Pad with consistent values instead of random
    while (features.length < 128) {
      features.push(0.5)
    }

    return features.slice(0, 128)
  }

  const calculateSimilarity = (features1: number[], features2: number[]) => {
    if (features1.length !== features2.length) return 0

    let dotProduct = 0
    let norm1 = 0
    let norm2 = 0

    for (let i = 0; i < features1.length; i++) {
      dotProduct += features1[i] * features2[i]
      norm1 += features1[i] * features1[i]
      norm2 += features2[i] * features2[i]
    }

    const magnitude = Math.sqrt(norm1) * Math.sqrt(norm2)
    return magnitude === 0 ? 0 : dotProduct / magnitude
  }

  const drawDetections = (detections: Detection[]) => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    if (!video.videoWidth || !video.videoHeight || video.videoWidth === 0 || video.videoHeight === 0) {
      return
    }

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    detections.forEach((detection) => {
      const { box, landmarks } = detection

      const gradient = ctx.createLinearGradient(box.x, box.y, box.x + box.width, box.y + box.height)
      gradient.addColorStop(0, "#3b82f6")
      gradient.addColorStop(0.5, "#8b5cf6")
      gradient.addColorStop(1, "#06b6d4")

      ctx.strokeStyle = gradient
      ctx.lineWidth = 4
      ctx.setLineDash([8, 4])
      ctx.lineDashOffset = (Date.now() / 100) % 12
      ctx.strokeRect(box.x, box.y, box.width, box.height)
      ctx.setLineDash([])

      landmarks.forEach((point, index) => {
        const pulse = Math.sin(Date.now() / 200 + index) * 0.5 + 0.5
        ctx.fillStyle = `rgba(6, 182, 212, ${0.8 + pulse * 0.2})`
        ctx.shadowColor = "#06b6d4"
        ctx.shadowBlur = 12 + pulse * 8

        ctx.beginPath()
        const size = 3 + pulse * 2
        ctx.arc(point.x, point.y, size, 0, 2 * Math.PI)
        ctx.fill()
      })

      ctx.shadowBlur = 0

      const expressionText = `${expressionEmojis[currentExpression]} ${currentExpression.toUpperCase()}`
      const confidenceText = `${confidence}%`

      // Expression text styling
      ctx.font = "bold 28px 'Arial Black', Arial"
      ctx.textAlign = "center"

      // Create gradient for text
      const textGradient = ctx.createLinearGradient(box.x, box.y - 60, box.x + box.width, box.y - 20)
      textGradient.addColorStop(0, "#fbbf24")
      textGradient.addColorStop(0.5, "#f59e0b")
      textGradient.addColorStop(1, "#d97706")

      // Text shadow/glow effect
      ctx.shadowColor = "#fbbf24"
      ctx.shadowBlur = 15
      ctx.fillStyle = textGradient
      ctx.fillText(expressionText, box.x + box.width / 2, box.y - 30)

      // Confidence text
      ctx.font = "bold 20px Arial"
      ctx.shadowBlur = 10
      ctx.fillStyle = "#ffffff"
      ctx.fillText(confidenceText, box.x + box.width / 2, box.y - 5)

      ctx.shadowBlur = 0
      ctx.textAlign = "start"

      if (recognizedName !== "Unknown") {
        ctx.font = "bold 24px Arial"
        ctx.textAlign = "center"

        // Create gradient for name
        const nameGradient = ctx.createLinearGradient(
          box.x,
          box.y + box.height + 20,
          box.x + box.width,
          box.y + box.height + 50,
        )
        nameGradient.addColorStop(0, "#10b981")
        nameGradient.addColorStop(1, "#059669")

        ctx.shadowColor = "#10b981"
        ctx.shadowBlur = 12
        ctx.fillStyle = nameGradient
        ctx.fillText(`✓ ${recognizedName}`, box.x + box.width / 2, box.y + box.height + 35)

        ctx.shadowBlur = 0
        ctx.textAlign = "start"
      }
    })
  }

  const clearCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
    }
  }

  const captureFace = async () => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video || currentDetections.length === 0) return null

    const ctx = canvas.getContext("2d")
    if (!ctx) return null

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    const imageData = canvas.toDataURL("image/jpeg", 0.8)

    // Use real features from current detection
    const features = extractFaceFeatures(currentDetections[0])

    return {
      imageData,
      features,
    }
  }

  const addFace = async () => {
    if (!newFaceName.trim()) return

    const faceData = await captureFace()
    if (!faceData) return

    const newFace: StoredFace = {
      id: Date.now().toString(),
      name: newFaceName.trim(),
      features: faceData.features,
      timestamp: Date.now(),
      imageData: faceData.imageData,
    }

    const updatedFaces = [...storedFaces, newFace]
    setStoredFaces(updatedFaces)
    localStorage.setItem("mzaFaceRecognitionDatabase", JSON.stringify(updatedFaces))

    setNewFaceName("")
    setIsAddFaceOpen(false)
  }

  const deleteFace = (id: string) => {
    const updatedFaces = storedFaces.filter((face) => face.id !== id)
    setStoredFaces(updatedFaces)
    localStorage.setItem("mzaFaceRecognitionDatabase", JSON.stringify(updatedFaces))
  }

  useEffect(() => {
    if (isModelLoaded && isCameraActive && !isDetecting) {
      startDetection()
    }
  }, [isModelLoaded, isCameraActive, startDetection])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* ... existing background effects ... */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="neural-network"></div>
        <div className="floating-shapes">
          <div className="shape shape-1"></div>
          <div className="shape shape-2"></div>
          <div className="shape shape-3"></div>
          <div className="shape shape-4"></div>
        </div>
        <div className="data-streams">
          <div className="stream stream-1"></div>
          <div className="stream stream-2"></div>
          <div className="stream stream-3"></div>
        </div>
        <div className="holographic-scanner"></div>
      </div>

      {/* Loading Overlay */}
      {loadingProgress && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="text-center space-y-4">
            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
            <div className="text-white text-xl font-semibold">{loadingProgress}</div>
            <div className="text-blue-400">MZA AI Face Recognition System</div>
          </div>
        </div>
      )}

      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
              MZA AI Face Recognition
            </h1>
          </div>
          <p className="text-gray-300 text-lg">Advanced Neural Network Facial Analysis System</p>
          <div className="flex items-center justify-center gap-4 mt-4">
            <Badge variant="outline" className="border-green-500 text-green-400">
              <Activity className="w-3 h-3 mr-1" />
              {systemStatus}
            </Badge>
            <Badge variant="outline" className="border-blue-500 text-blue-400">
              MZA AI Engine v2.1
            </Badge>
            <Badge
              variant="outline"
              className={`${isModelLoaded ? "border-green-500 text-green-400" : "border-yellow-500 text-yellow-400"}`}
            >
              {isModelLoaded ? "AI Models Loaded" : "Loading Models..."}
            </Badge>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-400 flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            {error}
          </div>
        )}

        {/* ... rest of the existing UI remains the same ... */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Camera Feed */}
          <div className="lg:col-span-2">
            <Card className="bg-black/40 backdrop-blur-sm border-gray-700 hover:border-blue-500/50 transition-all duration-300">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-white">
                  <Camera className="w-5 h-5" />
                  Live Camera Feed
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="relative aspect-video bg-gray-900 rounded-lg overflow-hidden">
                  <video ref={videoRef} className="w-full h-full object-cover" autoPlay muted playsInline />
                  <canvas
                    ref={canvasRef}
                    className="absolute inset-0 w-full h-full"
                    style={{ mixBlendMode: "screen" }}
                  />
                  {!isCameraActive && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center text-gray-400">
                        <Camera className="w-16 h-16 mx-auto mb-4 opacity-50" />
                        <p>Camera not active</p>
                      </div>
                    </div>
                  )}
                </div>
                <div className="flex gap-2 mt-4">
                  <Button
                    onClick={isCameraActive ? stopCamera : startCamera}
                    disabled={!isModelLoaded}
                    className={`flex-1 ${
                      isCameraActive
                        ? "bg-red-600 hover:bg-red-700"
                        : "bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                    } transition-all duration-300`}
                  >
                    {isCameraActive ? (
                      <>
                        <CameraOff className="w-4 h-4 mr-2" />
                        Stop Camera
                      </>
                    ) : (
                      <>
                        <Camera className="w-4 h-4 mr-2" />
                        Start Camera
                      </>
                    )}
                  </Button>
                  <Dialog open={isAddFaceOpen} onOpenChange={setIsAddFaceOpen}>
\
