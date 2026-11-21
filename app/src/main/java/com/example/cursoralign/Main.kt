package com.example.cursoralign

import android.app.Activity
import android.app.Application
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.Point
import android.graphics.Rect
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Process
import android.provider.Settings
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import rikka.shizuku.Shizuku
import java.nio.ByteBuffer
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.hypot

class MainApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        Prefs.init(this)
        try { OpenCVLoader.initLocal() } catch (_: Exception) { }
    }
}

object AppState {
    var enabled = false
    var triggerActive = false
    var targetPoint: Point? = null
    var screenW = 0
    var screenH = 0
    var cursorX = 0
    var cursorY = 0
    var floatingServiceRunning = false
    var screenCaptureServiceRunning = false
    var overlayView: OverlayView? = null
}

object Config {
    const val PROCESS_WIDTH = 640
    const val FACE_ENABLED = true
    const val FACE_MIN_BOX = 36
    const val TARGET_TEMPLATE_THRESHOLD = 0.62
    const val ALIGN_CONFIDENCE = 0.60
    const val TRIGGER_THRESHOLD = 0.70
    const val TRIGGER_POS_TOL = 16
    const val TRIGGER_SCALE_TOL = 0.12f
    const val CURSOR_THRESHOLD = 0.75
    const val DEADZONE = 22
    const val SWIPE_INTERVAL_MS = 240L
    const val SWIPE_DURATION_MS = 110
    const val SWIPE_FACTOR = 0.55f
    const val SWIPE_MODE_AWAY_FROM_TARGET = true
    const val TEMPLATE_EVERY_N_FRAMES = 2
    const val PAIR_MATCH_THRESHOLD = 0.45
    const val BODY_CONF_FACTOR = 0.85
    const val BODY_DEFAULT_TOP = 0.0
    const val BODY_DEFAULT_HEIGHT = 0.10
    const val BODY_DEFAULT_WIDTH = 0.45
    const val BODY_DEFAULT_CENTER_X = 0.5
    val TRIGGER_ROI: Rect? = null
}

object Prefs {
    private const val PREF_NAME = "cursor_align_prefs"
    private const val KEY_ENABLED = "cursor_align_enabled"
    private lateinit var prefs: SharedPreferences
    fun init(context: Context) { prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE) }
    fun isEnabled(): Boolean = prefs.getBoolean(KEY_ENABLED, false)
    fun setEnabled(value: Boolean) { prefs.edit().putBoolean(KEY_ENABLED, value).apply() }
}

object ShizukuInput {
    fun swipe(x1: Int, y1: Int, x2: Int, y2: Int, durationMs: Int) {
        if (!Shizuku.pingBinder()) return
        val command = "input swipe $x1 $y1 $x2 $y2 $durationMs"
        Thread {
            try { Shizuku.newProcess(arrayOf("sh", "-c", command), null, null).waitFor() } catch (_: Exception) { }
        }.start()
    }
}

data class MatchResult(val value: Double, val x: Int, val y: Int, val w: Int, val h: Int, val scaleUsed: Double)

data class Box(val left: Int, val top: Int, val right: Int, val bottom: Int, val conf: Double, val source: String) {
    val cx: Int get() = (left + right) / 2
    val cy: Int get() = (top + bottom) / 2
}

data class FrameResult(
    val triggerActive: Boolean,
    val candidates: List<Box>,
    val selected: Box?,
    val bodies: List<Box>
)

class DetectionEngine(private val context: Context) {

    private val targetTemplates = mutableListOf<Mat>()
    private val bodyTemplates = mutableListOf<Mat>()
    private val triggerTemplates = mutableListOf<Pair<String, Mat>>()
    private var cursorTemplate: Mat? = null
    private var loaded = false

    private var learnedTop = Config.BODY_DEFAULT_TOP
    private var learnedHeight = Config.BODY_DEFAULT_HEIGHT
    private var learnedWidth = Config.BODY_DEFAULT_WIDTH
    private var learnedCenterX = Config.BODY_DEFAULT_CENTER_X

    private var refTriggerCx = -1.0
    private var refTriggerCy = -1.0
    private var refTriggerScale = 1.0

    private var lastTemplateHeads: List<Box> = emptyList()
    private var lastBodies: List<Box> = emptyList()

    private val faceOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .build()
    private val faceDetector = FaceDetection.getClient(faceOptions)

    fun ensureLoaded() { if (loaded) return; loadTemplates(); loaded = true }

    private fun loadTemplates() {
        releaseTemplates()
        try {
            val files = context.assets.list("detection") ?: emptyArray<String>()

            val headMap = mutableMapOf<Int, Mat>()
            val bodyMap = mutableMapOf<Int, Mat>()

            files.filter { it.startsWith("target_b") }.forEach { name ->
                val n = name.removePrefix("target_b").substringBefore(".").toIntOrNull() ?: return@forEach
                loadGrayMat("detection/$name")?.let { headMap[n] = it; targetTemplates.add(it) }
            }

            files.filter { it.startsWith("body_b") }.forEach { name ->
                val n = name.removePrefix("body_b").substringBefore(".").toIntOrNull() ?: return@forEach
                loadGrayMat("detection/$name")?.let { bodyMap[n] = it; bodyTemplates.add(it) }
            }

            files.filter { it.startsWith("trigger_active") }.sorted().forEach { name ->
                loadGrayMat("detection/$name")?.let { triggerTemplates.add(Pair("active", it)) }
            }

            val normal = loadGrayMat("detection/trigger_normal.jpg") ?: loadGrayMat("detection/trigger_normal.png")
            if (normal != null) triggerTemplates.add(0, Pair("normal", normal))

            cursorTemplate = loadGrayMat("detection/cursor_a.jpg") ?: loadGrayMat("detection/cursor_a.png")

            learnHeadRatio(headMap, bodyMap)
        } catch (_: Exception) { }
    }

    private fun learnHeadRatio(headMap: Map<Int, Mat>, bodyMap: Map<Int, Mat>) {
        var sumTop = 0.0
        var sumHeight = 0.0
        var sumWidth = 0.0
        var sumCenterX = 0.0
        var count = 0

        for ((n, bodyMat) in bodyMap) {
            val headMat = headMap[n] ?: continue
            val m = matchTemplate(bodyMat, headMat, 1.0, 1.0)
            if (m != null && m.value >= Config.PAIR_MATCH_THRESHOLD) {
                val bw = bodyMat.cols().toDouble()
                val bh = bodyMat.rows().toDouble()
                sumTop += m.y / bh
                sumHeight += m.h / bh
                sumWidth += m.w / bw
                sumCenterX += (m.x + m.w / 2.0) / bw
                count++
            }
        }

        if (count > 0) {
            learnedTop = sumTop / count
            learnedHeight = sumHeight / count
            learnedWidth = sumWidth / count
            learnedCenterX = sumCenterX / count
        } else {
            learnedTop = Config.BODY_DEFAULT_TOP
            learnedHeight = Config.BODY_DEFAULT_HEIGHT
            learnedWidth = Config.BODY_DEFAULT_WIDTH
            learnedCenterX = Config.BODY_DEFAULT_CENTER_X
        }
    }

    private fun loadGrayMat(path: String): Mat? {
        return try {
            val bitmap = BitmapFactory.decodeStream(context.assets.open(path)) ?: return null
            val rgba = Mat()
            Utils.bitmapToMat(bitmap, rgba)
            val gray = Mat()
            if (rgba.channels() == 4) Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            else if (rgba.channels() == 3) Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGB2GRAY)
            else rgba.copyTo(gray)
            rgba.release()
            bitmap.recycle()
            gray
        } catch (_: Exception) { null }
    }

    fun processFull(bitmap: Bitmap, screenW: Int, screenH: Int, frameIndex: Int): FrameResult {
        ensureLoaded()
        if (bitmap.width <= 0 || bitmap.height <= 0) return FrameResult(false, emptyList(), null, emptyList())

        AppState.cursorX = screenW / 2
        AppState.cursorY = screenH / 2

        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val scale = Config.PROCESS_WIDTH.toDouble() / screenW.toDouble()
        val processH = (screenH * scale).toInt().coerceAtLeast(1)
        val smallRgba = Mat()
        Imgproc.resize(rgba, smallRgba, Size(Config.PROCESS_WIDTH.toDouble(), processH.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val gray = Mat()
        if (smallRgba.channels() == 4) Imgproc.cvtColor(smallRgba, gray, Imgproc.COLOR_RGBA2GRAY)
        else if (smallRgba.channels() == 3) Imgproc.cvtColor(smallRgba, gray, Imgproc.COLOR_RGB2GRAY)
        else smallRgba.copyTo(gray)
        rgba.release()

        val candidates = mutableListOf<Box>()

        if (Config.FACE_ENABLED) {
            candidates.addAll(detectFaces(smallRgba, scale, screenW, screenH))
        }

        if (frameIndex % Config.TEMPLATE_EVERY_N_FRAMES == 0) {
            val r = detectTemplateTargets(gray, scale, screenW, screenH)
            lastTemplateHeads = r.first
            lastBodies = r.second
        }
        candidates.addAll(lastTemplateHeads)

        val triggerActive = detectTrigger(gray, scale)

        refineCursor(gray, scale, screenW, screenH)

        val merged = dedupe(candidates)
        val selected = merged.minByOrNull { hypot((it.cx - AppState.cursorX).toDouble(), (it.cy - AppState.cursorY).toDouble()) }

        gray.release()
        smallRgba.release()

        return FrameResult(triggerActive, merged, selected, lastBodies)
    }

    private fun detectFaces(smallRgba: Mat, scale: Double, screenW: Int, screenH: Int): List<Box> {
        return try {
            val smallBmp = Bitmap.createBitmap(smallRgba.cols(), smallRgba.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(smallRgba, smallBmp)
            val image = InputImage.fromBitmap(smallBmp, 0)
            val faces = Tasks.await(faceDetector.process(image), 400, TimeUnit.MILLISECONDS)
            val out = mutableListOf<Box>()
            for (f in faces) {
                val b = f.boundingBox
                if (b.width() >= Config.FACE_MIN_BOX && b.height() >= Config.FACE_MIN_BOX) {
                    out.add(Box(
                        (b.left / scale).toInt().coerceIn(0, screenW),
                        (b.top / scale).toInt().coerceIn(0, screenH),
                        (b.right / scale).toInt().coerceIn(0, screenW),
                        (b.bottom / scale).toInt().coerceIn(0, screenH),
                        0.95, "face"))
                }
            }
            smallBmp.recycle()
            out
        } catch (_: Exception) { emptyList() }
    }

    private fun detectTemplateTargets(gray: Mat, scale: Double, screenW: Int, screenH: Int): Pair<List<Box>, List<Box>> {
        val heads = mutableListOf<Box>()
        val bodies = mutableListOf<Box>()

        for (t in targetTemplates) {
            val m = matchTemplate(gray, t, scale, 1.0)
            if (m != null && m.value >= Config.TARGET_TEMPLATE_THRESHOLD) {
                heads.add(boxFromMatch(m, scale, screenW, screenH, m.value, "template"))
            }
        }

        for (t in bodyTemplates) {
            val m = matchTemplate(gray, t, scale, 1.0)
            if (m != null && m.value >= Config.TARGET_TEMPLATE_THRESHOLD) {
                val full = boxFromMatch(m, scale, screenW, screenH, m.value, "body")
                bodies.add(full)
                heads.add(headBoxFromBody(full))
            }
        }

        return Pair(heads, bodies)
    }

    private fun headBoxFromBody(b: Box): Box {
        val w = (b.right - b.left).toDouble()
        val hgt = (b.bottom - b.top).toDouble()
        val headTop = b.top + (hgt * learnedTop).toInt()
        val headH = (hgt * learnedHeight).toInt().coerceAtLeast(8)
        val headW = (w * learnedWidth).toInt().coerceAtLeast(8)
        val headCx = b.left + (w * learnedCenterX).toInt()
        return Box(
            headCx - headW / 2,
            headTop,
            headCx + headW / 2,
            headTop + headH,
            b.conf * Config.BODY_CONF_FACTOR,
            "body-head")
    }

    private fun detectTrigger(gray: Mat, scale: Double): Boolean {
        if (triggerTemplates.isEmpty()) return false

        var bestName = ""
        var bestMatch: MatchResult? = null
        var bestScale = 1.0
        val scales = doubleArrayOf(0.8, 0.9, 1.0, 1.1, 1.2)

        for (pair in triggerTemplates) {
            for (s in scales) {
                val m = matchTemplate(gray, pair.second, scale, s)
                if (m != null && m.value >= Config.TRIGGER_THRESHOLD) {
                    if (bestMatch == null || m.value > bestMatch.value) {
                        bestMatch = m
                        bestName = pair.first
                        bestScale = s
                    }
                }
            }
        }

        val m = bestMatch ?: return false
        val cx = (m.x + m.w / 2.0) / scale
        val cy = (m.y + m.h / 2.0) / scale

        if (bestName == "active") return true

        if (refTriggerCx < 0) {
            refTriggerCx = cx
            refTriggerCy = cy
            refTriggerScale = bestScale
            return false
        }

        val posDelta = hypot(cx - refTriggerCx, cy - refTriggerCy)
        val scaleDelta = abs(bestScale - refTriggerScale)
        return posDelta > Config.TRIGGER_POS_TOL || scaleDelta > Config.TRIGGER_SCALE_TOL
    }

    private fun refineCursor(gray: Mat, scale: Double, screenW: Int, screenH: Int) {
        val t = cursorTemplate ?: return
        val roiHalf = (250 * scale).toInt()
        val cxSmall = (AppState.cursorX * scale).toInt()
        val cySmall = (AppState.cursorY * scale).toInt()
        val l = (cxSmall - roiHalf).coerceIn(0, gray.cols() - 1)
        val tp = (cySmall - roiHalf).coerceIn(0, gray.rows() - 1)
        val r = (cxSmall + roiHalf).coerceIn(l + 1, gray.cols())
        val bt = (cySmall + roiHalf).coerceIn(tp + 1, gray.rows())
        if (r - l < 30 || bt - tp < 30) return
        val sub = gray.submat(tp, bt, l, r)
        val m = matchTemplate(sub, t, scale, 1.0)
        if (m != null && m.value >= Config.CURSOR_THRESHOLD) {
            AppState.cursorX = ((l + m.x + m.w / 2.0) / scale).toInt().coerceIn(0, screenW)
            AppState.cursorY = ((tp + m.y + m.h / 2.0) / scale).toInt().coerceIn(0, screenH)
        }
        sub.release()
    }

    private fun boxFromMatch(m: MatchResult, scale: Double, screenW: Int, screenH: Int, conf: Double, source: String): Box {
        return Box(
            (m.x / scale).toInt().coerceIn(0, screenW),
            (m.y / scale).toInt().coerceIn(0, screenH),
            ((m.x + m.w) / scale).toInt().coerceIn(0, screenW),
            ((m.y + m.h) / scale).toInt().coerceIn(0, screenH),
            conf, source)
    }

    private fun dedupe(boxes: List<Box>): List<Box> {
        val sorted = boxes.sortedByDescending { it.conf }
        val kept = mutableListOf<Box>()
        for (b in sorted) {
            var dup = false
            for (k in kept) {
                if (hypot((b.cx - k.cx).toDouble(), (b.cy - k.cy).toDouble()) < 70) { dup = true; break }
            }
            if (!dup) kept.add(b)
        }
        return kept
    }

    private fun matchTemplate(search: Mat, template: Mat, captureScale: Double, extraScale: Double): MatchResult? {
        if (search.empty() || template.empty()) return null
        val tw = (template.cols() * captureScale * extraScale).toInt()
        val th = (template.rows() * captureScale * extraScale).toInt()
        if (tw < 12 || th < 12) return null
        if (tw >= search.cols() || th >= search.rows()) return null
        val resized = Mat()
        Imgproc.resize(template, resized, Size(tw.toDouble(), th.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val result = Mat()
        return try {
            Imgproc.matchTemplate(search, resized, result, Imgproc.TM_CCOEFF_NORMED)
            val mm = Core.minMaxLoc(result)
            MatchResult(mm.maxVal, mm.maxLoc.x.toInt(), mm.maxLoc.y.toInt(), tw, th, extraScale)
        } catch (_: Exception) { null } finally { resized.release(); result.release() }
    }

    private fun releaseTemplates() {
        targetTemplates.forEach { it.release() }
        targetTemplates.clear()
        bodyTemplates.forEach { it.release() }
        bodyTemplates.clear()
        triggerTemplates.forEach { it.second.release() }
        triggerTemplates.clear()
        cursorTemplate?.release()
        cursorTemplate = null
        refTriggerCx = -1.0
        refTriggerCy = -1.0
        refTriggerScale = 1.0
    }
}
class OverlayView(context: Context) : View(context) {

    private val boxPaint = Paint().apply { color = Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 4f }
    private val bodyPaint = Paint().apply { color = Color.CYAN; style = Paint.Style.STROKE; strokeWidth = 3f }
    private val targetPaint = Paint().apply { color = Color.RED; style = Paint.Style.STROKE; strokeWidth = 6f }
    private val linePaint = Paint().apply { color = Color.YELLOW; style = Paint.Style.STROKE; strokeWidth = 5f }
    private val crossPaint = Paint().apply { color = Color.YELLOW; style = Paint.Style.STROKE; strokeWidth = 4f }
    private val textPaint = Paint().apply { color = Color.WHITE; textSize = 38f; setShadowLayer(8f, 0f, 0f, Color.BLACK) }

    @Volatile private var boxes: List<Box> = emptyList()
    @Volatile private var bodies: List<Box> = emptyList()
    @Volatile private var target: Box? = null
    @Volatile private var cursorX = 0
    @Volatile private var cursorY = 0
    @Volatile private var status = ""

    fun update(boxes: List<Box>, bodies: List<Box>, target: Box?, cx: Int, cy: Int, status: String) {
        this.boxes = boxes
        this.bodies = bodies
        this.target = target
        this.cursorX = cx
        this.cursorY = cy
        this.status = status
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (b in bodies) {
            canvas.drawRect(b.left.toFloat(), b.top.toFloat(), b.right.toFloat(), b.bottom.toFloat(), bodyPaint)
        }

        for (b in boxes) {
            canvas.drawRect(b.left.toFloat(), b.top.toFloat(), b.right.toFloat(), b.bottom.toFloat(), boxPaint)
        }

        target?.let { t ->
            canvas.drawRect(t.left.toFloat(), t.top.toFloat(), t.right.toFloat(), t.bottom.toFloat(), targetPaint)
            canvas.drawLine(cursorX.toFloat(), cursorY.toFloat(), t.cx.toFloat(), t.cy.toFloat(), linePaint)
            canvas.drawCircle(t.cx.toFloat(), t.cy.toFloat(), 14f, targetPaint)
        }

        canvas.drawLine(cursorX - 30f, cursorY.toFloat(), cursorX + 30f, cursorY.toFloat(), crossPaint)
        canvas.drawLine(cursorX.toFloat(), cursorY - 30f, cursorX.toFloat(), cursorY + 30f, crossPaint)
        canvas.drawText(status, 24f, 60f, textPaint)
    }
}

class MainActivity : Activity() {

    companion object {
        private const val OVERLAY_REQ = 1001
        private const val MEDIA_REQ = 1002
        private const val NOTIF_REQ = 1003
        private const val SHIZUKU_REQ = 1004
    }

    private lateinit var btn: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = FrameLayout(this)
        btn = Button(this)
        btn.text = "Start"
        btn.setOnClickListener {
            if (AppState.floatingServiceRunning || AppState.screenCaptureServiceRunning) stopAll() else startAll()
        }
        root.addView(btn, FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.CENTER))
        setContentView(root)
        updateBtn()
    }

    override fun onResume() { super.onResume(); updateBtn() }

    private fun updateBtn() {
        btn.text = if (AppState.floatingServiceRunning || AppState.screenCaptureServiceRunning) "Stop" else "Start"
    }

    private fun startAll() {
        if (!Settings.canDrawOverlays(this)) {
            Toast.makeText(this, "Overlay permission do", Toast.LENGTH_LONG).show()
            startActivityForResult(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, android.net.Uri.parse("package:$packageName")), OVERLAY_REQ)
            return
        }
        if (!Shizuku.pingBinder()) { Toast.makeText(this, "Shizuku start karo", Toast.LENGTH_LONG).show(); return }
        if (Shizuku.checkSelfPermission() != PackageManager.PERMISSION_GRANTED) { Shizuku.requestPermission(SHIZUKU_REQ); return }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(android.Manifest.permission.POST_NOTIFICATIONS), NOTIF_REQ)
            return
        }
        startActivityForResult((getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager).createScreenCaptureIntent(), MEDIA_REQ)
    }

    private fun stopAll() {
        AppState.floatingServiceRunning = false
        AppState.screenCaptureServiceRunning = false
        stopService(Intent(this, FloatingService::class.java))
        stopService(Intent(this, ScreenCaptureService::class.java))
        updateBtn()
    }

    override fun onActivityResult(req: Int, res: Int, data: Intent?) {
        super.onActivityResult(req, res, data)
        if (req == OVERLAY_REQ && Settings.canDrawOverlays(this)) startAll()
        if (req == MEDIA_REQ && res == RESULT_OK && data != null) {
            AppState.screenCaptureServiceRunning = true
            val i = Intent(this, ScreenCaptureService::class.java).putExtra(ScreenCaptureService.EXTRA_DATA, data)
            startForegroundService(i)
            startForegroundService(Intent(this, FloatingService::class.java))
            AppState.floatingServiceRunning = true
            updateBtn()
        }
    }

    override fun onRequestPermissionsResult(req: Int, perms: Array<out String>, res: IntArray) {
        super.onRequestPermissionsResult(req, perms, res)
        Shizuku.onRequestPermissionResult(req, res)
        if (req == SHIZUKU_REQ && res.isNotEmpty() && res[0] == PackageManager.PERMISSION_GRANTED) startAll()
        if (req == NOTIF_REQ && res.isNotEmpty() && res[0] == PackageManager.PERMISSION_GRANTED) startAll()
    }
}
class FloatingService : Service() {

    private lateinit var wm: WindowManager
    private var iconView: View? = null
    private var menuView: View? = null
    private var overlay: OverlayView? = null

    override fun onBind(i: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        AppState.floatingServiceRunning = true
        AppState.enabled = Prefs.isEnabled()
        wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager

        val overlayType = if (Build.VERSION.SDK_INT >= 26) WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY else WindowManager.LayoutParams.TYPE_PHONE

        overlay = OverlayView(this)
        val op = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT,
            overlayType,
            WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT)
        op.gravity = Gravity.TOP or Gravity.START
        wm.addView(overlay, op)
        AppState.overlayView = overlay

        val iconFrame = FrameLayout(this)
        val iv = ImageView(this)
        iv.scaleType = ImageView.ScaleType.CENTER_CROP
        loadIcon(iv)
        val closeBtn = TextView(this)
        closeBtn.text = "X"
        closeBtn.setTextColor(Color.WHITE)
        closeBtn.gravity = Gravity.CENTER
        closeBtn.setBackgroundColor(Color.RED)
        closeBtn.setOnClickListener { stopSelf(); Process.killProcess(Process.myPid()) }
        iv.setOnClickListener { iconView?.visibility = View.GONE; menuView?.visibility = View.VISIBLE }
        iconFrame.addView(iv, FrameLayout.LayoutParams(150, 150))
        iconFrame.addView(closeBtn, FrameLayout.LayoutParams(60, 60, Gravity.TOP or Gravity.END))
        val p = WindowManager.LayoutParams(WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.WRAP_CONTENT, overlayType, WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE, PixelFormat.TRANSLUCENT)
        p.gravity = Gravity.TOP or Gravity.START
        p.x = 50
        p.y = 200
        wm.addView(iconFrame, p)
        iconView = iconFrame

        val menu = LinearLayout(this)
        menu.orientation = LinearLayout.VERTICAL
        menu.setBackgroundColor(Color.argb(220, 30, 30, 30))
        menu.setPadding(30, 30, 30, 30)
        val title = TextView(this)
        title.text = "Cursor Align"
        title.setTextColor(Color.WHITE)
        val menuClose = TextView(this)
        menuClose.text = "X"
        menuClose.setTextColor(Color.WHITE)
        menuClose.gravity = Gravity.CENTER
        menuClose.setBackgroundColor(Color.RED)
        menuClose.setOnClickListener { menuView?.visibility = View.GONE; iconView?.visibility = View.VISIBLE }
        val header = LinearLayout(this)
        header.addView(title, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        header.addView(menuClose, LinearLayout.LayoutParams(80, 80))
        val row = LinearLayout(this)
        val lbl = TextView(this)
        lbl.text = "Cursor Align Target"
        lbl.setTextColor(Color.WHITE)
        val sw = Switch(this)
        sw.isChecked = AppState.enabled
        sw.setOnCheckedChangeListener { _, isChecked ->
            AppState.enabled = isChecked
            Prefs.setEnabled(isChecked)
            if (!isChecked) { AppState.triggerActive = false; AppState.targetPoint = null }
        }
        row.addView(lbl, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        row.addView(sw)
        menu.addView(header)
        menu.addView(row)
        val mp = WindowManager.LayoutParams(600, WindowManager.LayoutParams.WRAP_CONTENT, overlayType, WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE, PixelFormat.TRANSLUCENT)
        mp.gravity = Gravity.TOP or Gravity.START
        mp.x = 50
        mp.y = 200
        menu.visibility = View.GONE
        wm.addView(menu, mp)
        menuView = menu

        getSystemService(NotificationManager::class.java).createNotificationChannel(NotificationChannel("CH1", "Float", NotificationManager.IMPORTANCE_LOW))
        val notification = Notification.Builder(this, "CH1").setContentTitle("Cursor Align").setContentText("Running").setSmallIcon(android.R.drawable.ic_menu_info_details).build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(2001, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(2001, notification)
        }
    }

    private fun loadIcon(iv: ImageView) {
        val jpg = try { BitmapFactory.decodeStream(assets.open("floating_icon.jpg")) } catch (_: Exception) { null }
        if (jpg != null) { iv.setImageBitmap(jpg); return }
        val png = try { BitmapFactory.decodeStream(assets.open("floating_icon.png")) } catch (_: Exception) { null }
        if (png != null) { iv.setImageBitmap(png); return }
        iv.setImageResource(android.R.drawable.ic_menu_gallery)
    }

    override fun onDestroy() {
        AppState.floatingServiceRunning = false
        AppState.overlayView = null
        try {
            iconView?.let { wm.removeView(it) }
            menuView?.let { wm.removeView(it) }
            overlay?.let { wm.removeView(it) }
        } catch (_: Exception) { }
        iconView = null
        menuView = null
        overlay = null
        super.onDestroy()
    }
}
class ScreenCaptureService : Service() {

    companion object {
        const val EXTRA_DATA = "extra_media_projection_data"
    }

    private var mp: MediaProjection? = null
    private var vd: VirtualDisplay? = null
    private var ir: ImageReader? = null
    private val thread = HandlerThread("CaptureThread")
    private lateinit var h: Handler
    private var engine: DetectionEngine? = null
    private var lastSwipe = 0L
    private var frameIndex = 0

    override fun onBind(i: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        AppState.screenCaptureServiceRunning = true
        thread.start()
        h = Handler(thread.looper)
        engine = DetectionEngine(this)
    }

    override fun onStartCommand(i: Intent?, f: Int, s: Int): Int {
        getSystemService(NotificationManager::class.java).createNotificationChannel(NotificationChannel("CH2", "Cap", NotificationManager.IMPORTANCE_LOW))
        val notification = Notification.Builder(this, "CH2").setContentTitle("Cursor Align").setContentText("Capturing").setSmallIcon(android.R.drawable.ic_menu_camera).build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(2002, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION)
        } else {
            startForeground(2002, notification)
        }

        val data: Intent? = if (Build.VERSION.SDK_INT >= 33) {
            i?.getParcelableExtra(EXTRA_DATA, Intent::class.java)
        } else {
            (i?.getParcelableExtra(EXTRA_DATA) as Intent?)
        }
        if (data == null) { stopSelf(); return START_NOT_STICKY }

        mp = (getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager).getMediaProjection(Activity.RESULT_OK, data)
        val wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val p = Point()
        wm.defaultDisplay.getRealSize(p)
        AppState.screenW = p.x
        AppState.screenH = p.y
        AppState.cursorX = p.x / 2
        AppState.cursorY = p.y / 2
        ir = ImageReader.newInstance(p.x, p.y, PixelFormat.RGBA_8888, 2)
        vd = mp?.createVirtualDisplay("S", p.x, p.y, resources.configuration.densityDpi, DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, ir!!.surface, null, h)
        h.postDelayed({ loop() }, 150)
        return START_STICKY
    }

    private fun loop() {
        if (!AppState.screenCaptureServiceRunning) return
        val img = ir?.acquireLatestImage()
        if (img != null) {
            val bmp = imageToBitmap(img)
            img.close()
            if (bmp != null) {
                try {
                    frameIndex++
                    val res = engine?.processFull(bmp, AppState.screenW, AppState.screenH, frameIndex)
                    if (res != null) handleResult(res)
                } catch (_: Exception) { } finally { bmp.recycle() }
            }
        }
        h.postDelayed({ loop() }, 120)
    }

    private fun handleResult(res: FrameResult) {
        AppState.triggerActive = res.triggerActive
        AppState.targetPoint = res.selected?.let { Point(it.cx, it.cy) }

        val status = "EN:" + (if (AppState.enabled) "ON" else "OFF") +
                " TRIG:" + (if (res.triggerActive) "YES" else "NO") +
                " HEAD:" + res.candidates.size +
                " BODY:" + res.bodies.size +
                (res.selected?.let { " CONF:" + String.format("%.2f", it.conf) } ?: "")

        AppState.overlayView?.update(res.candidates, res.bodies, res.selected, AppState.cursorX, AppState.cursorY, status)

        if (!AppState.enabled || !res.triggerActive) return
        val t = res.selected ?: return
        if (t.conf < Config.ALIGN_CONFIDENCE) return

        val dx = t.cx - AppState.cursorX
        val dy = t.cy - AppState.cursorY
        if (abs(dx) <= Config.DEADZONE && abs(dy) <= Config.DEADZONE) return

        val now = System.currentTimeMillis()
        if (now - lastSwipe < Config.SWIPE_INTERVAL_MS) return
        lastSwipe = now

        var sx = AppState.cursorX
        var sy = AppState.cursorY
        var ex = sx
        var ey = sy
        if (Config.SWIPE_MODE_AWAY_FROM_TARGET) {
            ex = (sx - dx * Config.SWIPE_FACTOR).toInt()
            ey = (sy - dy * Config.SWIPE_FACTOR).toInt()
        } else {
            sx = t.cx
            sy = t.cy
            ex = AppState.cursorX
            ey = AppState.cursorY
        }

        ShizukuInput.swipe(
            sx.coerceIn(0, AppState.screenW),
            sy.coerceIn(0, AppState.screenH),
            ex.coerceIn(0, AppState.screenW),
            ey.coerceIn(0, AppState.screenH),
            Config.SWIPE_DURATION_MS)
    }

    private fun imageToBitmap(image: Image): Bitmap? {
        return try {
            val plane = image.planes[0]
            val buf = plane.buffer
            val w = image.width
            val hgt = image.height
            val bmp = Bitmap.createBitmap(w, hgt, Bitmap.Config.ARGB_8888)
            if (plane.pixelStride == 4 && plane.rowStride == w * 4) {
                bmp.copyPixelsFromBuffer(buf)
                return bmp
            }
            val rgba = ByteArray(hgt * w * 4)
            for (y in 0 until hgt) {
                val rowStart = y * plane.rowStride
                for (x in 0 until w) {
                    val si = rowStart + x * plane.pixelStride
                    val di = (y * w + x) * 4
                    rgba[di] = buf.get(si)
                    rgba[di + 1] = buf.get(si + 1)
                    rgba[di + 2] = buf.get(si + 2)
                    rgba[di + 3] = buf.get(si + 3)
                }
            }
            bmp.copyPixelsFromBuffer(ByteBuffer.wrap(rgba))
            bmp
        } catch (_: Exception) { null }
    }

    override fun onDestroy() {
        AppState.screenCaptureServiceRunning = false
        try {
            h.removeCallbacksAndMessages(null)
            vd?.release()
            ir?.close()
            mp?.stop()
            thread.quitSafely()
        } catch (_: Exception) { }
        vd = null
        ir = null
        mp = null
        super.onDestroy()
    }
}
