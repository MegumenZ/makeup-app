import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import {
  Camera,
  ShoppingBag,
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  Check,
  Star,
  Menu,
  X,
  Loader2,
  Sparkles,
  Upload,
  RefreshCcw,
  Scan,
  PlayCircle,
} from "lucide-react";

// Import dataset JSON
import makeupDataRaw from "./makeup_data.json";

// Placeholder Image (Base64)
const PLACEHOLDER_IMAGE =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300' viewBox='0 0 300 300'%3E%3Crect width='300' height='300' fill='%23f0f0f0'/%3E%3Ctext x='50%25' y='50%25' font-family='sans-serif' font-size='20' fill='%23999' dominant-baseline='middle' text-anchor='middle'%3ENo Image Available%3C/text%3E%3C/svg%3E";

// --- KONFIGURASI WARNA KULIT ---
const SKIN_TONE_PALETTES = {
  0: {
    title: "Deep Cool",
    desc: "Undertone dingin yang kaya, cocok dengan warna gelap intens dan deep cocoa.",
    targetColors: [
      "#5D3A28",
      "#8B0000",
      "#4B2E2A",
      "#3E2723",
      "#581845",
      "#6D271A",
    ],
  },
  1: {
    title: "Warm Medium",
    desc: "Undertone kuning langsat/sawo matang dengan nuansa hangat, honey, dan tan.",
    targetColors: [
      "#D29C7B",
      "#B35A5A",
      "#C68642",
      "#CD853F",
      "#A56B57",
      "#D2691E",
    ],
  },
  2: {
    title: "Fair Ivory",
    desc: "Warna kulit terang dengan undertone netral atau pinkish porcelain.",
    targetColors: [
      "#F5E0D6",
      "#FFE4C4",
      "#FFDEAD",
      "#FAEBD7",
      "#F0E68C",
      "#FFC0CB",
    ],
  },
};

// --- ALGORITMA ---
const hexToRgb = (hex) => {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : null;
};

const calculateColorDistance = (hex1, hex2) => {
  const rgb1 = hexToRgb(hex1);
  const rgb2 = hexToRgb(hex2);
  if (!rgb1 || !rgb2) return 1000;

  return Math.sqrt(
    Math.pow(rgb1.r - rgb2.r, 2) +
      Math.pow(rgb1.g - rgb2.g, 2) +
      Math.pow(rgb1.b - rgb2.b, 2)
  );
};

export default function TrueShadeApp() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);

  const [model, setModel] = useState(null);
  const [modelLoading, setModelLoading] = useState(true);
  const [modelError, setModelError] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);

  const [recommendedProducts, setRecommendedProducts] = useState([]);

  // STATE PAGINATION & ANIMATION
  const [currentPage, setCurrentPage] = useState(1);
  const [slideDirection, setSlideDirection] = useState("next"); // 'next' or 'prev'
  const itemsPerPage = 4;

  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [stream, setStream] = useState(null);

  const fileInputRef = useRef(null);
  const imageRef = useRef(null);
  const videoRef = useRef(null);
  const productsRef = useRef(null);

  // --- 1. LOAD MODEL ---
  useEffect(() => {
    const loadModel = async () => {
      try {
        setModelLoading(true);
        const response = await fetch("/model/model.json");
        if (!response.ok) throw new Error("File model tidak ditemukan.");
        const modelJSON = await response.json();

        if (modelJSON?.modelTopology?.model_config?.config?.layers) {
          modelJSON.modelTopology.model_config.config.layers.forEach(
            (layer) => {
              if (
                layer.class_name === "InputLayer" &&
                layer.config.batch_shape
              ) {
                layer.config.batch_input_shape = layer.config.batch_shape;
                delete layer.config.batch_shape;
              }
            }
          );
        }

        const blob = new Blob([JSON.stringify(modelJSON)], {
          type: "application/json",
        });
        const patchedModelUrl = URL.createObjectURL(blob);
        const loadedModel = await tf.loadLayersModel(patchedModelUrl, {
          weightPathPrefix: "/model/",
        });

        loadedModel.predict(tf.zeros([1, 224, 224, 3])).dispose();
        setModel(loadedModel);
        setModelLoading(false);
      } catch (error) {
        console.error("ERROR LOAD MODEL:", error);
        setModelError(error.message);
        setModelLoading(false);
      }
    };
    loadModel();
  }, []);

  // --- 1.5 CLEANUP CAMERA ---
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [stream]);

  // --- 2. KAMERA & UPLOAD ---
  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
      setStream(mediaStream);
      setIsCameraOpen(true);
      setPreviewImage(null);
      setAnalysisComplete(false);
      setTimeout(() => {
        if (videoRef.current) videoRef.current.srcObject = mediaStream;
      }, 100);
    } catch (err) {
      console.error("Error akses kamera:", err);
      alert("Gagal mengakses kamera. Pastikan izin diberikan.");
    }
  };

  const stopCamera = () => {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    setStream(null);
    setIsCameraOpen(false);
  };

  const capturePhoto = () => {
    if (videoRef.current) {
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      setPreviewImage(canvas.toDataURL("image/jpeg", 0.9));
      stopCamera();
      startAnalysis();
    }
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setPreviewImage(URL.createObjectURL(file));
      setAnalysisComplete(false);
      setPredictionResult(null);
      startAnalysis();
    }
  };

  // --- 3. REKOMENDASI (NO LIMIT) ---
  const getProductRecommendations = (skinClassIndex) => {
    if (!makeupDataRaw || !Array.isArray(makeupDataRaw)) return [];

    const skinPalette = SKIN_TONE_PALETTES[skinClassIndex];
    const targetColors = skinPalette.targetColors;
    let scoredProducts = [];

    makeupDataRaw.forEach((product) => {
      let bestMatchScore = 1000;
      let matchedShadeName = "";
      let matchedHex = "";

      if (product.product_colors && Array.isArray(product.product_colors)) {
        product.product_colors.forEach((prodColor) => {
          targetColors.forEach((skinHex) => {
            const distance = calculateColorDistance(
              prodColor.hex_value,
              skinHex
            );
            if (distance < 60) {
              if (distance < bestMatchScore) {
                bestMatchScore = distance;
                matchedShadeName = prodColor.colour_name;
                matchedHex = prodColor.hex_value;
              }
            }
          });
        });
      }

      if (bestMatchScore < 60) {
        scoredProducts.push({
          ...product,
          matchPercent: Math.max(80, Math.round(100 - bestMatchScore)),
          bestShade: matchedShadeName,
          bestHex: matchedHex,
        });
      }
    });

    // Return ALL matching products (sorted)
    return scoredProducts.sort((a, b) => b.matchPercent - a.matchPercent);
  };

  // --- 4. ANALISIS ---
  const startAnalysis = async () => {
    setIsAnalyzing(true);
    setRecommendedProducts([]);
    setCurrentPage(1);

    setTimeout(async () => {
      if (model && imageRef.current) {
        try {
          const tensor = tf.browser
            .fromPixels(imageRef.current)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(tf.scalar(255.0))
            .expandDims();

          const predictions = await model.predict(tensor).data();
          const maxIndex = predictions.indexOf(Math.max(...predictions));

          setPredictionResult(maxIndex);
          const recs = getProductRecommendations(maxIndex);
          setRecommendedProducts(recs);

          tensor.dispose();
        } catch (error) {
          console.error("Error prediksi:", error);
          alert("Gagal memproses gambar.");
        }
      }
      setIsAnalyzing(false);
      setAnalysisComplete(true);
    }, 1500);
  };

  // --- 5. LOGIKA PAGINATION & SCROLL ---
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentProducts = recommendedProducts.slice(
    indexOfFirstItem,
    indexOfLastItem
  );
  const totalPages = Math.ceil(recommendedProducts.length / itemsPerPage);

  const smartScrollToTop = () => {
    // Hanya scroll di mobile/tablet vertical
    if (window.innerWidth < 768 && productsRef.current) {
      productsRef.current.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  };

  const nextPage = () => {
    if (currentPage < totalPages) {
      setSlideDirection("next"); // Set arah slide ke kiri
      setCurrentPage(currentPage + 1);
      smartScrollToTop();
    }
  };

  const prevPage = () => {
    if (currentPage > 1) {
      setSlideDirection("prev"); // Set arah slide ke kanan
      setCurrentPage(currentPage - 1);
      smartScrollToTop();
    }
  };

  const goToPage = (pageNumber) => {
    setSlideDirection(pageNumber > currentPage ? "next" : "prev");
    setCurrentPage(pageNumber);
    smartScrollToTop();
  };

  const getPageNumbers = () => {
    const pageNumbers = [];
    const maxButtons = 5;
    if (totalPages <= maxButtons) {
      for (let i = 1; i <= totalPages; i++) pageNumbers.push(i);
    } else {
      let startPage = Math.max(1, currentPage - Math.floor(maxButtons / 2));
      let endPage = Math.min(totalPages, startPage + maxButtons - 1);
      if (endPage - startPage + 1 < maxButtons)
        startPage = Math.max(1, endPage - maxButtons + 1);
      for (let i = startPage; i <= endPage; i++) pageNumbers.push(i);
    }
    return pageNumbers;
  };

  const resultData =
    predictionResult !== null
      ? SKIN_TONE_PALETTES[predictionResult]
      : SKIN_TONE_PALETTES[1];

  return (
    <div className="min-h-screen w-full overflow-x-hidden bg-stone-50 font-sans text-stone-800">
      {/* STYLE UNTUK ANIMASI SLIDE */}
      <style>{`
        @keyframes slideInRight {
          from { opacity: 0; transform: translateX(50px); }
          to { opacity: 1; transform: translateX(0); }
        }
        @keyframes slideInLeft {
          from { opacity: 0; transform: translateX(-50px); }
          to { opacity: 1; transform: translateX(0); }
        }
        .animate-slide-next {
          animation: slideInRight 0.4s ease-out forwards;
        }
        .animate-slide-prev {
          animation: slideInLeft 0.4s ease-out forwards;
        }
      `}</style>

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleImageUpload}
        className="hidden"
        accept="image/*"
      />

      {/* NAVBAR */}
      <nav className="fixed top-0 w-full bg-white/80 backdrop-blur-md z-50 border-b border-stone-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex-shrink-0 flex items-center gap-2 cursor-pointer">
              <div className="w-8 h-8 bg-rose-400 rounded-full flex items-center justify-center text-white">
                <Sparkles size={18} />
              </div>
              <span className="font-serif text-xl font-bold tracking-tight">
                TrueShade
              </span>
            </div>
            <div className="hidden md:flex space-x-8 text-sm font-medium text-stone-600">
              <a href="#" className="hover:text-rose-500 transition">
                Beranda
              </a>
              <a href="#analyzer" className="hover:text-rose-500 transition">
                Analisis Wajah
              </a>
              <a href="#products" className="hover:text-rose-500 transition">
                Produk
              </a>
            </div>
            <div className="flex items-center gap-4">
              <button className="p-2 hover:bg-stone-100 rounded-full relative">
                <ShoppingBag size={20} />
                {recommendedProducts.length > 0 && (
                  <span className="absolute top-1 right-1 w-2 h-2 bg-rose-500 rounded-full"></span>
                )}
              </button>
              <button
                className="md:hidden p-2 hover:bg-stone-100 rounded-full"
                onClick={() => setIsMenuOpen(!isMenuOpen)}
              >
                {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* HERO SECTION */}
      <header className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 overflow-hidden w-full">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div className="text-center md:text-left space-y-6">
              <span className="inline-block py-1 px-3 rounded-full bg-rose-100 text-rose-800 text-xs font-semibold tracking-wide uppercase">
                AI Powered Technology
              </span>
              <h1 className="text-4xl md:text-6xl font-serif font-bold leading-tight text-stone-900">
                Temukan Shade <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-rose-400 to-orange-400">
                  Sempurna Kamu
                </span>
              </h1>
              <p className="text-lg text-stone-600 md:max-w-md mx-auto md:mx-0 leading-relaxed">
                Unggah foto dan biarkan AI mencocokkan warna kulitmu dengan
                ribuan produk kosmetik nyata.
              </p>

              {/* BUTTONS */}
              <div className="flex flex-col sm:flex-row gap-4 justify-center md:justify-start pt-4">
                <button
                  onClick={() =>
                    document
                      .getElementById("analyzer")
                      .scrollIntoView({ behavior: "smooth" })
                  }
                  className="px-8 py-4 bg-stone-900 text-white rounded-full font-medium hover:bg-rose-500 transition-all duration-300 flex items-center justify-center gap-2 shadow-lg shadow-rose-200"
                >
                  <Camera size={20} />
                  Coba Analisis Sekarang
                </button>
                <button
                  onClick={() =>
                    document
                      .getElementById("how-it-works")
                      .scrollIntoView({ behavior: "smooth" })
                  }
                  className="px-8 py-4 bg-white text-stone-800 border border-stone-200 rounded-full font-medium hover:bg-stone-50 transition-all duration-300 flex items-center justify-center gap-2 shadow-sm hover:shadow-md"
                >
                  <PlayCircle size={20} />
                  Cara Kerja
                </button>
              </div>
            </div>
            <div className="relative">
              <div className="absolute -inset-4 bg-gradient-to-tr from-rose-200 to-orange-100 rounded-full blur-3xl opacity-40 animate-pulse"></div>
              <img
                src="https://images.unsplash.com/photo-1616683693504-3ea7e9ad6fec?auto=format&fit=crop&q=80&w=800"
                alt="Model Makeup"
                className="relative rounded-[2rem] shadow-2xl w-full object-cover h-[400px] md:h-[500px]"
                onError={(e) => (e.target.src = PLACEHOLDER_IMAGE)}
              />
            </div>
          </div>
        </div>
      </header>

      {/* HOW IT WORKS SECTION */}
      <section
        id="how-it-works"
        className="py-20 bg-white border-b border-stone-100"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="font-serif text-3xl font-bold text-stone-900">
              Cara Kerja
            </h2>
            <p className="text-stone-500 mt-2">
              Tiga langkah mudah menuju penampilan sempurna.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
            <div className="flex flex-col items-center text-center group">
              <div className="w-20 h-20 bg-rose-50 rounded-full flex items-center justify-center text-rose-500 mb-6 group-hover:scale-110 transition-transform duration-300 shadow-sm">
                <Upload size={32} />
              </div>
              <h3 className="text-xl font-bold text-stone-900 mb-2">
                1. Upload Foto
              </h3>
              <p className="text-stone-500 leading-relaxed px-4">
                Ambil selfie langsung atau unggah foto close-up wajahmu dari
                galeri.
              </p>
            </div>
            <div className="flex flex-col items-center text-center group">
              <div className="w-20 h-20 bg-rose-50 rounded-full flex items-center justify-center text-rose-500 mb-6 group-hover:scale-110 transition-transform duration-300 shadow-sm">
                <Scan size={32} />
              </div>
              <h3 className="text-xl font-bold text-stone-900 mb-2">
                2. Analisis AI
              </h3>
              <p className="text-stone-500 leading-relaxed px-4">
                Sistem cerdas kami mendeteksi undertone dan warna kulitmu dalam
                hitungan detik.
              </p>
            </div>
            <div className="flex flex-col items-center text-center group">
              <div className="w-20 h-20 bg-rose-50 rounded-full flex items-center justify-center text-rose-500 mb-6 group-hover:scale-110 transition-transform duration-300 shadow-sm">
                <ShoppingBag size={32} />
              </div>
              <h3 className="text-xl font-bold text-stone-900 mb-2">
                3. Temukan Produk
              </h3>
              <p className="text-stone-500 leading-relaxed px-4">
                Dapatkan rekomendasi kosmetik yang warnanya dijamin cocok
                untukmu.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ANALYZER SECTION (FIXED MOBILE LAYOUT) */}
      <section
        id="analyzer"
        className="py-20 bg-stone-900 text-white overflow-hidden relative"
      >
        <div className="absolute top-0 left-0 w-full h-full opacity-10 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')]"></div>
        <div className="max-w-4xl mx-auto px-4 relative z-10">
          <div className="text-center mb-10">
            <h2 className="font-serif text-3xl md:text-4xl font-bold mb-4">
              Analisis AI
            </h2>
            <div className="flex flex-col items-center justify-center gap-3">
              {modelLoading && (
                <span className="text-yellow-400 flex items-center gap-1 text-sm bg-yellow-400/10 px-3 py-1 rounded-full">
                  <Loader2 size={14} className="animate-spin" /> Memuat Model
                  AI...
                </span>
              )}
              {!modelLoading && !modelError && (
                <span className="text-green-400 flex items-center gap-1 text-sm bg-green-400/10 px-3 py-1 rounded-full">
                  <Check size={14} /> Sistem AI Siap
                </span>
              )}
            </div>
          </div>

          <div className="bg-stone-800 rounded-3xl overflow-hidden shadow-2xl border border-stone-700">
            <div className="flex flex-col md:grid md:grid-cols-2 min-h-[500px]">
              {/* KIRI: INPUT AREA */}
              <div className="relative bg-black flex flex-col items-center justify-center p-6 overflow-hidden h-[400px] md:h-auto w-full order-1">
                {!previewImage && !isCameraOpen && (
                  <>
                    <div className="w-24 h-24 md:w-32 md:h-32 rounded-full border-4 border-stone-600 border-dashed flex items-center justify-center mb-6">
                      <Camera
                        size={32}
                        className="text-stone-500 md:w-10 md:h-10"
                      />
                    </div>
                    <div className="flex flex-col gap-3 w-full max-w-xs z-10">
                      <button
                        onClick={startCamera}
                        disabled={modelLoading}
                        className="w-full bg-rose-500 hover:bg-rose-600 text-white px-6 py-3 rounded-full font-medium transition flex items-center justify-center gap-2 shadow-lg"
                      >
                        <Camera size={18} /> Ambil Foto
                      </button>
                      <button
                        onClick={() => fileInputRef.current.click()}
                        disabled={modelLoading}
                        className="w-full bg-stone-700 hover:bg-stone-600 text-white px-6 py-3 rounded-full font-medium transition flex items-center justify-center gap-2 shadow-lg"
                      >
                        <Upload size={18} /> Upload Galeri
                      </button>
                    </div>
                  </>
                )}

                {isCameraOpen && (
                  <div className="absolute inset-0 flex flex-col bg-black">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="flex-1 w-full h-full object-cover transform scale-x-[-1]"
                    />
                    <div className="absolute bottom-6 left-0 right-0 flex justify-center gap-4 z-20">
                      <button
                        onClick={stopCamera}
                        className="p-4 bg-white/20 backdrop-blur-sm rounded-full text-white"
                      >
                        <X size={24} />
                      </button>
                      <button
                        onClick={capturePhoto}
                        className="w-16 h-16 bg-white rounded-full border-4 border-rose-500 flex items-center justify-center"
                      >
                        <div className="w-12 h-12 bg-rose-500 rounded-full"></div>
                      </button>
                    </div>
                  </div>
                )}

                {previewImage && (
                  <div className="relative w-full h-full flex items-center justify-center bg-black">
                    <img
                      ref={imageRef}
                      src={previewImage}
                      alt="Preview"
                      className={`w-full h-full object-contain ${
                        isAnalyzing ? "opacity-50 blur-sm" : ""
                      } transition-all`}
                      crossOrigin="anonymous"
                    />
                    {isAnalyzing && (
                      <div className="absolute inset-0 flex flex-col items-center justify-center z-20">
                        <Loader2
                          size={48}
                          className="animate-spin text-rose-500 mb-2"
                        />
                        <p className="text-rose-200 bg-black/50 px-3 py-1 rounded-full backdrop-blur-sm">
                          Menganalisis...
                        </p>
                      </div>
                    )}
                    {!isAnalyzing && (
                      <button
                        onClick={() => {
                          setPreviewImage(null);
                          setAnalysisComplete(false);
                        }}
                        className="absolute bottom-4 right-4 bg-black/60 text-white p-2 rounded-full flex items-center gap-2 px-4 border border-white/20 hover:bg-black/80 backdrop-blur-sm z-20"
                      >
                        <RefreshCcw size={16} /> Ulangi
                      </button>
                    )}
                  </div>
                )}
              </div>

              {/* KANAN: HASIL AREA */}
              <div className="p-6 md:p-8 flex flex-col justify-center bg-stone-800 w-full min-h-[300px] md:min-h-auto order-2">
                {!analysisComplete ? (
                  <div className="space-y-4 opacity-30 text-center">
                    <Sparkles
                      size={48}
                      className="mx-auto mb-4 text-stone-600"
                    />
                    <p className="text-stone-500">
                      Hasil analisis warna kulitmu akan muncul di sini.
                    </p>
                  </div>
                ) : (
                  <div className="animate-fade-in">
                    <span className="text-rose-400 text-sm font-bold tracking-wider uppercase mb-2 block">
                      Tone Terdeteksi
                    </span>
                    <h3 className="text-3xl font-serif font-bold mb-1 text-white">
                      {resultData.title}
                    </h3>
                    <p className="text-stone-400 text-sm mb-6">
                      {resultData.desc}
                    </p>

                    <div className="pt-4 border-t border-stone-700">
                      <p className="text-xs text-stone-500 mb-2 uppercase tracking-wide">
                        Palette Cocok:
                      </p>
                      <div className="flex gap-3 mb-6 flex-wrap">
                        {resultData.targetColors.slice(0, 5).map((color, i) => (
                          <div
                            key={i}
                            className="w-8 h-8 rounded-full border border-white/20 shadow-sm"
                            style={{ backgroundColor: color }}
                          ></div>
                        ))}
                      </div>
                      <button
                        onClick={() =>
                          document
                            .getElementById("products")
                            .scrollIntoView({ behavior: "smooth" })
                        }
                        className="w-full bg-white text-stone-900 py-3 rounded-lg font-bold hover:bg-stone-200 transition flex items-center justify-center gap-2 shadow-lg"
                      >
                        Lihat Rekomendasi Produk <ArrowRight size={16} />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* PRODUCT SECTION */}
      <section id="products" className="py-24 bg-stone-50" ref={productsRef}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="mb-12 flex flex-col md:flex-row justify-between items-end gap-4">
            <div>
              <h2 className="font-serif text-3xl font-bold text-stone-900">
                Rekomendasi Untukmu
              </h2>
              <p className="text-stone-500 mt-2">
                {analysisComplete
                  ? `Ditemukan ${recommendedProducts.length} produk cocok untuk tone "${resultData.title}"`
                  : "Lakukan analisis wajah untuk melihat produk yang dipersonalisasi."}
              </p>
            </div>

            {analysisComplete && recommendedProducts.length > 0 && (
              <div className="hidden md:flex items-center gap-2 text-sm text-stone-500 bg-white px-4 py-2 rounded-full border border-stone-200 shadow-sm">
                <span>
                  Halaman {currentPage} dari {totalPages}
                </span>
              </div>
            )}
          </div>

          {!analysisComplete && (
            <div className="flex flex-col items-center justify-center py-20 opacity-50 bg-white rounded-xl border border-dashed border-stone-300">
              <ShoppingBag size={48} className="mb-4 text-stone-300" />
              <p className="text-stone-400">
                Produk akan muncul di sini setelah analisis.
              </p>
            </div>
          )}

          {analysisComplete && recommendedProducts.length === 0 && (
            <div className="text-center py-10 text-stone-500 bg-white rounded-xl border border-stone-200">
              Maaf, tidak ditemukan produk yang sangat mirip di database saat
              ini.
            </div>
          )}

          {/* GRID PRODUK - DENGAN ANIMASI SLIDE */}
          {analysisComplete && recommendedProducts.length > 0 && (
            <>
              <div
                key={currentPage}
                className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 ${
                  slideDirection === "next"
                    ? "animate-slide-next"
                    : "animate-slide-prev"
                }`}
              >
                {currentProducts.map((product) => (
                  <div
                    key={product.id}
                    className="bg-white rounded-xl overflow-hidden shadow-sm hover:shadow-xl transition-all duration-300 group flex flex-col h-full border border-stone-100"
                  >
                    <div className="relative h-64 overflow-hidden bg-white p-4 flex items-center justify-center">
                      <img
                        src={product.image_link}
                        alt={product.name}
                        className="max-h-full max-w-full object-contain group-hover:scale-105 transition-transform duration-500"
                        onError={(e) => {
                          e.target.onerror = null;
                          e.target.src = PLACEHOLDER_IMAGE;
                        }}
                      />
                      <div className="absolute top-3 right-3 bg-white/90 backdrop-blur text-stone-900 text-xs font-bold py-1 px-2 rounded shadow-sm flex items-center gap-1">
                        <Star
                          size={12}
                          className="text-yellow-500 fill-yellow-500"
                        />
                        {product.matchPercent}% Match
                      </div>
                    </div>
                    <div className="p-5 flex-1 flex flex-col">
                      <p className="text-xs text-stone-500 mb-1 uppercase tracking-wider">
                        {product.brand || "Brand"}
                      </p>
                      <h3
                        className="font-bold text-stone-900 text-lg leading-tight mb-2 line-clamp-2"
                        title={product.name}
                      >
                        {product.name}
                      </h3>
                      <div className="flex items-center gap-2 mb-4 mt-auto">
                        <div
                          className="w-6 h-6 rounded-full border border-stone-200 shadow-sm"
                          style={{ backgroundColor: product.bestHex }}
                        ></div>
                        <div className="flex flex-col">
                          <span className="text-[10px] text-stone-400 uppercase">
                            Shade
                          </span>
                          <span className="text-xs font-semibold text-stone-700 line-clamp-1">
                            {product.bestShade}
                          </span>
                        </div>
                      </div>
                      <div className="pt-4 border-t border-stone-100 flex items-center justify-between">
                        <span className="font-serif font-bold text-rose-500">
                          {product.price_sign || "$"}
                          {product.price || "0.00"}
                        </span>
                        <a
                          href={product.product_link}
                          target="_blank"
                          rel="noreferrer"
                          className="w-8 h-8 rounded-full bg-stone-900 text-white flex items-center justify-center hover:bg-rose-500 transition"
                        >
                          <ShoppingBag size={14} />
                        </a>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* PAGINATION CONTROLS */}
              {totalPages > 1 && (
                <div className="flex justify-center items-center mt-12 w-full">
                  <div className="flex items-center gap-4">
                    <button
                      onClick={prevPage}
                      disabled={currentPage === 1}
                      className={`p-3 rounded-full border transition flex items-center justify-center ${
                        currentPage === 1
                          ? "bg-stone-100 text-stone-300 border-stone-200 cursor-not-allowed"
                          : "bg-white text-stone-800 border-stone-300 hover:bg-stone-50 hover:border-stone-400 shadow-sm"
                      }`}
                    >
                      <ChevronLeft size={20} />
                    </button>

                    <span className="text-sm font-semibold text-stone-600 md:hidden whitespace-nowrap">
                      Hal {currentPage} / {totalPages}
                    </span>

                    <div className="hidden md:flex gap-2">
                      {getPageNumbers().map((pageNum) => (
                        <button
                          key={pageNum}
                          onClick={() => goToPage(pageNum)}
                          className={`w-10 h-10 rounded-full text-sm font-bold transition flex-shrink-0 flex items-center justify-center ${
                            currentPage === pageNum
                              ? "bg-stone-900 text-white shadow-md"
                              : "bg-white text-stone-600 border border-stone-200 hover:bg-stone-50"
                          }`}
                        >
                          {pageNum}
                        </button>
                      ))}
                    </div>

                    <button
                      onClick={nextPage}
                      disabled={currentPage === totalPages}
                      className={`p-3 rounded-full border transition flex items-center justify-center ${
                        currentPage === totalPages
                          ? "bg-stone-100 text-stone-300 border-stone-200 cursor-not-allowed"
                          : "bg-white text-stone-800 border-stone-300 hover:bg-stone-50 hover:border-stone-400 shadow-sm"
                      }`}
                    >
                      <ChevronRight size={20} />
                    </button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </section>

      <footer className="bg-stone-900 text-stone-400 py-12 border-t border-stone-800">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-xs">
            &copy; 2024 TrueShade AI Inc. Data provided by Makeup API.
          </p>
        </div>
      </footer>
    </div>
  );
}
