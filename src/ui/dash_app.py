from __future__ import annotations

import base64
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Callable

import numpy as np

from dash import Dash, dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc


DATA_URL_RE = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)


def _safe_person_id_from_name(name: str) -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"\s+", "_", name)
    # латиница/кириллица/цифры/_/-
    name = re.sub(r"[^\w\u0400-\u04FF-]+", "", name, flags=re.UNICODE)
    return name or "unknown"


def _decode_data_url_to_bytes(data_url: str) -> bytes:
    if not data_url:
        raise ValueError("Пустой кадр")
    b64 = DATA_URL_RE.sub("", data_url)
    return base64.b64decode(b64)


def create_dash_app(
    *,
    fastapi_base_path: str = "/",
    init_system: Callable[[], Any],
    decode_image_bytes: Callable[[bytes], np.ndarray],
    detect_face_fn: Callable[[np.ndarray], Optional[tuple]],
    detect_face_info_fn: Callable[[np.ndarray], Optional[dict]],
    database_path: str,
) -> Dash:
    """
    Dash UI для вебкамеры:
    - запуск/остановка камеры в браузере
    - запоминание ТОЛЬКО multi-view по имени (одна кнопка)
    - авто-распознавание (каждые 2 секунды)
    """

    dash_app = Dash(
        __name__,
        requests_pathname_prefix=fastapi_base_path,
        routes_pathname_prefix="/",
        # Bootstrap оставляем для базовых компонентов, но визуально всё переопределяем CSS-ом (glass UI).
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
        title="Skin Recognition (Dash)",
        suppress_callback_exceptions=True,
    )

    # Полноэкранный layout (камера как фон) + glass панели поверх, как FaceID.
    dash_app.layout = html.Div(
        className="sr-root",
        children=[
            # realtime: по умолчанию ~10 FPS — регулируется слайдером
            dcc.Location(id="profile-location", refresh=False),
            dcc.Interval(id="recognize-interval", interval=100, disabled=True),
            # Always-on 3D tracking (facemesh + zones overlay). Saving happens only when enroll-store.active=True.
            dcc.Interval(id="enroll-interval", interval=240, disabled=False),
            dcc.Store(id="frame-store"),
            dcc.Store(id="bbox-store", data={"ok": False, "action": None, "ts": None, "frame": {"w": 0, "h": 0}, "mesh": {"points": []}, "mesh_n": 0, "_update_id": 0}),  # Инициализирован для первого срабатывания callback
            dcc.Store(id="enroll-store"),
            dcc.Store(id="recognize-enabled", data=False),
            html.Video(id="camera-video", className="sr-video", autoPlay=True, muted=True),
            html.Canvas(id="overlay-canvas", className="sr-overlay"),
            html.Canvas(id="camera-canvas", style={"display": "none"}),
            # Верхний HUD
            html.Div(
                className="sr-top glass",
                children=[
                    html.Div("Skin Recognition • исследовательский прототип", className="sr-title"),
                    html.Div(id="status-profile", className="sr-chip", children="Режим: проверка профиля…"),
                    html.Div(id="status-camera", className="sr-chip", children="Нажми «Камера» и разреши доступ."),
                    html.Div(id="status-mode", className="sr-chip", children="Распознавание: выключено"),
                    html.Div(id="status-recognition", className="sr-chip", children="Результат: —"),
                    html.Div(id="status-enroll", className="sr-chip", children="Ракурсы: активен"),
                ],
            ),
            # Панель управления (справа)
            html.Div(
                className="sr-panel glass",
                children=[
                    html.Div("Управление", className="sr-panel-title"),
                    html.Div(id="profile-description", className="sr-muted"),
                    html.Div(
                        className="sr-row",
                        children=[
                            html.Button("Камера", id="btn-start", className="sr-btn sr-btn-primary"),
                            html.Button("Стоп", id="btn-stop", className="sr-btn sr-btn-danger"),
                        ],
                    ),
                    dcc.Input(
                        id="person-name",
                        type="text",
                        placeholder="Имя для запоминания (например: Макс)",
                        className="sr-input",
                    ),
                    html.Button(
                        "Запомнить лицо",
                        id="btn-enroll-3d",
                        className="sr-btn sr-btn-accent sr-btn-wide",
                    ),
                    # progress bar (явная полоса, чтобы было видно, что идёт прогресс)
                    html.Div(
                        className="sr-progress-wrap",
                        children=[
                            html.Div("Прогресс регистрации", className="sr-label"),
                            dbc.Progress(
                                id="enroll-progress",
                                value=0,
                                max=100,
                                striped=True,
                                animated=True,
                                className="sr-progress",
                            ),
                        ],
                    ),
                    html.Button(
                        "Распознавать (вкл/выкл)",
                        id="btn-recognize-toggle",
                        className="sr-btn sr-btn-warn sr-btn-wide",
                    ),
                    html.Div(className="sr-label", children="Realtime FPS"),
                    dcc.Slider(
                        id="recognize-fps",
                        min=2,
                        max=15,
                        step=1,
                        value=10,
                        marks={2: "2", 5: "5", 10: "10", 15: "15"},
                        className="sr-slider",
                    ),
                    html.Div(id="fps-label", className="sr-muted"),
                    html.Div(id="result", className="sr-result"),
                ],
            ),
            html.Div(id="debug-clicks", className="sr-debug"),
            html.Div(id="debug-stream", className="sr-debug sr-debug-2"),
        ],
    )

    @dash_app.callback(
        Output("status-profile", "children"),
        Output("profile-description", "children"),
        Input("profile-location", "pathname"),
    )
    def describe_profile(pathname):
        try:
            stats = init_system().get_database_stats()
            scope = stats.get("recognition_scope", "skin_only")
            profiles = {
                "hybrid": ("Гибрид: лицо целиком + текстура кожи",
                           "Сравниваются признаки всего лица и локальная текстура кожи."),
                "whole_face": ("Лицо целиком", "Сравниваются признаки всего выровненного лица."),
                "whole_face_detector_only": ("Лицо целиком • обнаружение лица",
                                              "Сравниваются признаки всего лица; проверка участков кожи не применяется."),
                "skin_only": ("Текстура кожи", "Сравнивается локальная текстура выделенных участков кожи."),
            }
            label, description = profiles.get(scope, ("Неизвестный профиль", "Проверьте конфигурацию сервера."))
            return f"Режим: {label}", description
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Unable to display the active recognition profile")
            return "Режим: система недоступна", "Не удалось загрузить профиль. Подробности — в журнале сервера."

    # 1) Клиентская логика камеры (start/stop) — side-effect в браузере
    dash_app.clientside_callback(
        """
        function(nStart, nStop) {
            const video = document.getElementById("camera-video");
            if (!video) return "UI загружается...";

            function start() {
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    return "getUserMedia недоступен (нужен современный браузер и localhost/https)";
                }
                // Возвращаем Promise (Dash поддерживает async clientside callbacks)
                // Максимальное разрешение для распознавания по коже лица (поры, текстура)
                return navigator.mediaDevices.getUserMedia({
                    video: { 
                        width: { ideal: 1920, min: 1280 }, 
                        height: { ideal: 1080, min: 720 }, 
                        facingMode: "user",
                        frameRate: { ideal: 30, min: 15 }
                    }
                }).then((stream) => {
                    video.srcObject = stream;
                    return "Камера запущена (если нет картинки — проверь, что камера не занята другим приложением)";
                }).catch((e) => {
                    return "Ошибка доступа к камере: " + (e && e.message ? e.message : e);
                });
            }

            function stop() {
                try {
                    if (video.srcObject) {
                        const tracks = video.srcObject.getTracks();
                        tracks.forEach(t => t.stop());
                        video.srcObject = null;
                    }
                    return "Камера остановлена";
                } catch (e) {
                    return "Ошибка остановки камеры: " + (e && e.message ? e.message : e);
                }
            }

            // Не полагаемся на callback_context — определяем действие по счетчикам кликов
            const s = nStart || 0;
            const t = nStop || 0;
            if (s > t) {
                return start();
            }
            if (t >= s && t > 0) {
                return stop();
            }
            return dash_clientside.no_update;
        }
        """,
        Output("status-camera", "children"),
        Input("btn-start", "n_clicks"),
        Input("btn-stop", "n_clicks"),
        prevent_initial_call=True,
    )

    # 2) Toggle распознавания
    @dash_app.callback(
        Output("recognize-enabled", "data"),
        Output("recognize-interval", "disabled"),
        Output("status-mode", "children"),
        Input("btn-recognize-toggle", "n_clicks"),
        State("recognize-enabled", "data"),
        prevent_initial_call=True,
    )
    def toggle_recognize(n: int, enabled: bool):
        enabled = bool(enabled)
        new_enabled = not enabled
        if new_enabled:
            return True, False, "Распознавание: realtime (вкл)."
        return False, True, "Распознавание: выключено."

    @dash_app.callback(
        Output("recognize-interval", "interval"),
        Output("fps-label", "children"),
        Input("recognize-fps", "value"),
        prevent_initial_call=False,
    )
    def set_recognize_fps(fps: int):
        try:
            fps_i = int(fps)
        except Exception:
            fps_i = 10
        fps_i = max(2, min(15, fps_i))
        interval_ms = int(round(1000 / fps_i))
        return interval_ms, f"Текущая частота: ~{fps_i} FPS (интервал {interval_ms} ms)"

    # 2.5) Старт/стоп 3D enrollment
    @dash_app.callback(
        Output("enroll-store", "data", allow_duplicate=True),
        Output("enroll-interval", "disabled"),
        Output("status-enroll", "children", allow_duplicate=True),
        Input("btn-enroll-3d", "n_clicks"),
        State("person-name", "value"),
        prevent_initial_call=True,
    )
    def start_enroll(n: int, name: str):
        name = (name or "").strip()
        if not name:
            # 3D tracking всегда активен; без имени просто не начинаем сохранение.
            return no_update, False, "Ракурсы: активен (введите имя и нажмите «Запомнить», чтобы сохранить)"
        try:
            print(f"[dash] start_enroll: n={n} name='{name}'")
        except Exception:
            pass
        pid = _safe_person_id_from_name(name)
        store = {
            "active": True,
            "name": name,
            "person_id": pid,
            "need": {"center": 3, "left": 3, "right": 3, "up": 2, "down": 2},
            "got": {"center": 0, "left": 0, "right": 0, "up": 0, "down": 0},
        }
        return store, False, "Регистрация включена — смотрите прямо. Режим определяется по доступным ориентирам лица."

    # 3) Клиентская функция захвата кадра (БЕЗ callback_context, чтобы 100% работало)
    # 3.1) Always-on 3D tracking/enroll по enroll-interval
    dash_app.clientside_callback(
        """
        function(nEnrollTick, enrollStore, name) {
            try {
                window.__sr_capture = window.__sr_capture || { lastTs: 0, lastTick: 0, lastErr: "" };
                window.__sr_capture.lastTick = nEnrollTick || 0;

                const video = document.getElementById("camera-video");
                if (!video) {
                    window.__sr_capture.lastErr = "video element missing";
                    console.warn(`[enroll-capture] SKIP: video element missing`);
                    return dash_clientside.no_update;
                }
                // An unopened/stopped camera is a normal idle state.
                if (!video.srcObject) {
                    window.__sr_capture.lastErr = "";
                    return dash_clientside.no_update;
                }
                if (!video.videoWidth || !video.videoHeight) {
                    window.__sr_capture.lastErr = "video not ready (0x0)";
                    console.warn(`[enroll-capture] SKIP: video not ready (${video.videoWidth}x${video.videoHeight})`);
                    return dash_clientside.no_update;
                }

                const canvas = document.getElementById("camera-canvas");
                if (!canvas) {
                    window.__sr_capture.lastErr = "camera-canvas missing";
                    return dash_clientside.no_update;
                }

                // Для распознавания по коже лица нужно максимальное разрешение (поры, текстура)
                // Отправляем полное разрешение видео на сервер (без downscale)
                const vw = video.videoWidth;
                const vh = video.videoHeight;
                canvas.width = vw;
                canvas.height = vh;
                const canvasCtx = canvas.getContext("2d");
                canvasCtx.drawImage(video, 0, 0, vw, vh);
                // Высокое качество JPEG для сохранения деталей кожи
                const dataUrl = canvas.toDataURL("image/jpeg", 0.95);
                const meta = { w: canvas.width, h: canvas.height, vw: video.videoWidth, vh: video.videoHeight };

                window.__sr_capture.lastTs = Date.now();
                window.__sr_capture.lastErr = "";

                const action = (enrollStore && enrollStore.active) ? "enroll" : "track";
                const payload = { action: action, name: (enrollStore && enrollStore.name) || (name || ""), frame: dataUrl, meta: meta, ts: Date.now() };
                console.log(`[enroll-capture] RETURN: action=${action} frame_size=${dataUrl.length} ts=${payload.ts}`);
                return payload;
            } catch (e) {
                try {
                    window.__sr_capture = window.__sr_capture || { lastTs: 0, lastTick: 0, lastErr: "" };
                    window.__sr_capture.lastErr = "capture error: " + (e && e.message ? e.message : String(e));
                } catch (_) {}
                return dash_clientside.no_update;
            }
        }
        """,
        Output("frame-store", "data"),
        Input("enroll-interval", "n_intervals"),
        State("enroll-store", "data"),  # The next interval reads updates without a capture/store feedback loop.
        State("person-name", "value"),
        prevent_initial_call=False,  # False - callback срабатывает сразу
    )

    # 3.2) Realtime recognition по recognize-interval (когда включено)
    dash_app.clientside_callback(
        """
        function(nTick) {
            const video = document.getElementById("camera-video");
            if (!video || !video.videoWidth || !video.videoHeight) return dash_clientside.no_update;

            const canvas = document.getElementById("camera-canvas");
            // Для распознавания тоже нужно высокое разрешение (детали кожи)
            const vw = video.videoWidth;
            const vh = video.videoHeight;
            canvas.width = vw;
            canvas.height = vh;
            const canvasCtx = canvas.getContext("2d");
            canvasCtx.drawImage(video, 0, 0, vw, vh);
            const dataUrl = canvas.toDataURL("image/jpeg", 0.95);
            const meta = { w: canvas.width, h: canvas.height, vw: video.videoWidth, vh: video.videoHeight };

            return { action: "recognize", name: "", frame: dataUrl, meta: meta, ts: Date.now() };
        }
        """,
        Output("frame-store", "data", allow_duplicate=True),
        Input("recognize-interval", "n_intervals"),
        prevent_initial_call=True,
    )

    # 3.3) Client-side heartbeat: показывает, что тики идут и видео реально имеет размеры
    dash_app.clientside_callback(
        """
        function(nTick) {
            const video = document.getElementById("camera-video");
            if (!video) return "stream: video element missing";
            const hasStream = !!video.srcObject;
            const w = video.videoWidth || 0;
            const h = video.videoHeight || 0;
            const playing = (!video.paused && !video.ended);
            const cap = window.__sr_capture || { lastTs: 0, lastTick: 0, lastErr: "" };
            const ageMs = cap.lastTs ? (Date.now() - cap.lastTs) : 0;
            const age = cap.lastTs ? `${ageMs}ms` : "never";
            const err = cap.lastErr ? ` err="${cap.lastErr}"` : "";
            return `stream: tick=${nTick||0} hasStream=${hasStream} playing=${playing} video=${w}x${h} captureAge=${age} captureTick=${cap.lastTick||0}${err}`;
        }
        """,
        Output("debug-stream", "children"),
        Input("enroll-interval", "n_intervals"),
        prevent_initial_call=False,
    )

    # 3.5) Рисование bbox/лейбла поверх видео (YOLO-стайл)
    dash_app.clientside_callback(
        """
        function(bboxData, modeText, recText, fpsText) {
            // КРИТИЧЕСКАЯ ДИАГНОСТИКА: логируем ВСЁ в консоль браузера
            const isNull = (bboxData === null || bboxData === undefined);
            const bboxType = typeof bboxData;
            const bboxKeys = bboxData && typeof bboxData === "object" ? Object.keys(bboxData).join(",") : "N/A";
            console.log(`[bbox-draw] CALLBACK FIRED: isNull=${isNull} type=${bboxType} keys=${bboxKeys}`);
            if (!isNull) {
                console.log(`[bbox-draw] bboxData:`, JSON.stringify(bboxData).substring(0, 200));
            }
            
            try {
                const ok = bboxData && bboxData.ok;
                const meshN = bboxData && bboxData.mesh_n != null ? Number(bboxData.mesh_n) : (bboxData && bboxData.mesh && bboxData.mesh.points ? bboxData.mesh.points.length : 0);
                console.log(`[bbox-draw] parsed: isNull=${isNull} ok=${ok} mesh_n=${meshN}`);
            } catch (e) {
                console.error("[bbox-draw] parse error:", e);
            }

            const video = document.getElementById("camera-video");
            const overlay = document.getElementById("overlay-canvas");
            if (!video || !overlay) {
                console.warn("[bbox-draw] video or overlay missing");
                return dash_clientside.no_update;
            }

            // Canvas должен соответствовать РЕАЛЬНОМУ размеру элемента на экране,
            // иначе текст/графика масштабируются и "раздуваются" при fullscreen видео.
            const rect = video.getBoundingClientRect();
            const W = Math.max(1, Math.round(rect.width));
            const H = Math.max(1, Math.round(rect.height));
            const dpr = window.devicePixelRatio || 1;

            const targetW = Math.max(1, Math.round(W * dpr));
            const targetH = Math.max(1, Math.round(H * dpr));
            if (overlay.width !== targetW) overlay.width = targetW;
            if (overlay.height !== targetH) overlay.height = targetH;

            const ctx2 = overlay.getContext("2d");
            // Рисуем в CSS-пикселях (W/H), а не в device-пикселях
            ctx2.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx2.clearRect(0, 0, W, H);

            // realtime debug line (внизу слева) — НЕ прерываем отрисовку!
            let debug = "debug: —";
            try {
                const isNull = (bboxData === null || bboxData === undefined);
                const act = bboxData && bboxData.action ? String(bboxData.action) : "-";
                const ts = bboxData && bboxData.ts ? Number(bboxData.ts) : 0;
                const t = ts ? new Date(ts).toLocaleTimeString() : "";
                const ok = bboxData && bboxData.ok ? "ok" : "no";
                const meshN = bboxData && bboxData.mesh_n != null ? Number(bboxData.mesh_n) : (bboxData && bboxData.mesh && bboxData.mesh.points ? bboxData.mesh.points.length : 0);
                const face = bboxData && bboxData.bbox ? "face=1" : "face=0";
                const prog = bboxData && bboxData.enroll && bboxData.enroll.progress != null ? Math.round(Number(bboxData.enroll.progress) * 100) : 0;
                const wh = (bboxData && bboxData.frame && bboxData.frame.w && bboxData.frame.h) ? (String(bboxData.frame.w) + "x" + String(bboxData.frame.h)) : "";
                const keys = bboxData && typeof bboxData === "object" ? Object.keys(bboxData).join(",") : "null";
                debug = `t=${t} act=${act} ${ok} ${face} mesh=${meshN} prog=${prog}% ${wh} keys=[${keys}]`;
            } catch (e) {
                debug = "debug: error " + (e && e.message ? e.message : e);
            }

            if (!bboxData) {
                // bboxData is null/undefined - это НЕ нормально, значит Store не обновляется
                console.error(`[bbox-draw] ERROR: bboxData is null/undefined - Store не обновляется!`);
                return debug;
            }
            
            if (!bboxData.ok) {
                // ok=false - лицо не найдено или ошибка (это нормально)
                try {
                    const reason = !bboxData.ok ? "ok=false" : "unknown";
                    debug = debug + ` [SKIP:${reason}]`;
                    console.log(`[bbox-draw] SKIP: ${reason} (это нормально, если лицо не найдено)`, bboxData);
                } catch (_) {}
                return debug;
            }
            if (!bboxData.bbox || bboxData.bbox.w <= 0 || bboxData.bbox.h <= 0) {
                try {
                    debug = debug + ` [SKIP:bbox invalid]`;
                    console.warn(`[bbox-draw] SKIP: bbox invalid`, bboxData);
                } catch (_) {}
                return debug;
            }

            // Маппинг из координат КАДРА (0..1) в координаты ЭКРАНА при object-fit: cover.
            const frameW = (bboxData.frame && bboxData.frame.w) ? Number(bboxData.frame.w) : W;
            const frameH = (bboxData.frame && bboxData.frame.h) ? Number(bboxData.frame.h) : H;
            const scale = Math.max(W / frameW, H / frameH);
            const drawW = frameW * scale;
            const drawH = frameH * scale;
            const offX = (drawW - W) / 2.0;
            const offY = (drawH - H) / 2.0;
            function mapX(nx) { return (Number(nx || 0) * frameW * scale) - offX; }
            function mapY(ny) { return (Number(ny || 0) * frameH * scale) - offY; }

            const b = bboxData.bbox; // нормализованный bbox [0..1]
            const x = Math.round(mapX(b.x));
            const y = Math.round(mapY(b.y));
            const w = Math.round(Number(b.w || 0) * frameW * scale);
            const h = Math.round(Number(b.h || 0) * frameH * scale);

            // --- FaceMesh (как на вашем примере): точки + линии (без bbox-рамки) ---
            if (bboxData.mesh && bboxData.mesh.points && bboxData.mesh.points.length) {
                const pts = bboxData.mesh.points; // [[xn,yn], ...] в координатах КАДРА (0..1)
                const N = pts.length;
                // переводим в пиксели overlay
                const px = new Array(N);
                for (let i = 0; i < N; i++) {
                    const p = pts[i];
                    const xx = mapX(Math.max(0, Math.min(1, Number(p[0] || 0))));
                    const yy = mapY(Math.max(0, Math.min(1, Number(p[1] || 0))));
                    px[i] = [xx, yy];
                }

                // линии: k ближайших соседей -> плотная "как у facemesh"
                const K = 8; // плотнее сетка
                const maxD = Math.pow(Math.max(w, h) * 0.12, 2); // режем слишком дальние ребра, чтобы не было "паутин"
                ctx2.lineWidth = 2.0;
                ctx2.strokeStyle = "rgba(255,255,255,0.92)";
                ctx2.beginPath();
                for (let i = 0; i < N; i++) {
                    const xi = px[i][0], yi = px[i][1];
                    // выбираем K ближайших (простая O(N^2), но N~106 ок)
                    const bestD = new Array(K).fill(1e18);
                    const bestJ = new Array(K).fill(-1);
                    for (let j = 0; j < N; j++) {
                        if (j === i) continue;
                        const xj = px[j][0], yj = px[j][1];
                        const dx = xj - xi, dy = yj - yi;
                        const d = dx*dx + dy*dy;
                        // вставка в топ-K
                        for (let t = 0; t < K; t++) {
                            if (d < bestD[t]) {
                                for (let u = K - 1; u > t; u--) { bestD[u] = bestD[u-1]; bestJ[u] = bestJ[u-1]; }
                                bestD[t] = d; bestJ[t] = j;
                                break;
                            }
                        }
                    }
                    for (let t = 0; t < K; t++) {
                        const j = bestJ[t];
                        if (j < 0) continue;
                        if (bestD[t] > maxD) continue;
                        ctx2.moveTo(xi, yi);
                        ctx2.lineTo(px[j][0], px[j][1]);
                    }
                }
                ctx2.stroke();

                // точки
                ctx2.fillStyle = "rgba(255,255,255,0.98)";
                const r = 3.0;
                for (let i = 0; i < N; i++) {
                    ctx2.beginPath();
                    ctx2.arc(px[i][0], px[i][1], r, 0, Math.PI * 2);
                    ctx2.fill();
                }
            } else {
                // Haar supplies only a measured bounding box, not facial landmarks.
                ctx2.strokeStyle = "rgba(255,255,255,0.85)";
                ctx2.lineWidth = 2.0;
                ctx2.strokeRect(x, y, w, h);
            }

            // ---- Enrollment progress overlay ----
            if (bboxData && bboxData.enroll) {
                const e = bboxData.enroll;
                const hint = String(e.hint || "");

                // zone highlights inside bbox (shows that we collect "sections")
                const zones = (e.zones || {});
                function clamp01(v){ v = Number(v||0); return Math.max(0, Math.min(1, v)); }
                function fillRect(rx, ry, rw, rh, alpha, col){
                  if (alpha <= 0) return;
                  ctx2.fillStyle = col.replace("ALPHA", String(alpha.toFixed(3)));
                  ctx2.fillRect(rx, ry, rw, rh);
                }
                function meshRect(rx, ry, rw, rh, alpha) {
                  alpha = clamp01(alpha);
                  if (alpha <= 0) return;
                  const step = Math.max(10, Math.round(Math.min(rw, rh) / 12));
                  const r0 = 1.2;
                  ctx2.fillStyle = "rgba(255,255,255," + (0.35 * alpha).toFixed(3) + ")";
                  for (let yy = ry + step/2; yy < ry + rh; yy += step) {
                    for (let xx = rx + step/2; xx < rx + rw; xx += step) {
                      ctx2.beginPath();
                      ctx2.arc(xx, yy, r0, 0, Math.PI * 2);
                      ctx2.fill();
                    }
                  }
                }

                // define 5 regions relative to face bbox
                const leftA  = clamp01(zones.left);
                const rightA = clamp01(zones.right);
                const upA    = clamp01(zones.up);
                const downA  = clamp01(zones.down);
                const centerA= clamp01(zones.center);

                // semi-transparent glassy fills
                const rLeft   = {x: x + w*0.05, y: y + h*0.20, w: w*0.30, h: h*0.60};
                const rRight  = {x: x + w*0.65, y: y + h*0.20, w: w*0.30, h: h*0.60};
                const rUp     = {x: x + w*0.20, y: y + h*0.05, w: w*0.60, h: h*0.28};
                const rDown   = {x: x + w*0.20, y: y + h*0.67, w: w*0.60, h: h*0.28};
                const rCenter = {x: x + w*0.25, y: y + h*0.28, w: w*0.50, h: h*0.44};

                // подсветка зон (делаем заметнее)
                fillRect(rLeft.x, rLeft.y, rLeft.w, rLeft.h, Math.max(0.20, leftA) * 0.40, "rgba(0,194,255,ALPHA)");
                fillRect(rRight.x, rRight.y, rRight.w, rRight.h, Math.max(0.20, rightA) * 0.40, "rgba(0,194,255,ALPHA)");
                fillRect(rUp.x, rUp.y, rUp.w, rUp.h, Math.max(0.20, upA) * 0.40, "rgba(0,194,255,ALPHA)");
                fillRect(rDown.x, rDown.y, rDown.w, rDown.h, Math.max(0.20, downA) * 0.40, "rgba(0,194,255,ALPHA)");
                fillRect(rCenter.x, rCenter.y, rCenter.w, rCenter.h, Math.max(0.20, centerA) * 0.40, "rgba(0,194,255,ALPHA)");

                // "mesh" точки — показывают, что зона сохранена
                meshRect(rLeft.x, rLeft.y, rLeft.w, rLeft.h, leftA >= 1 ? 1 : 0);
                meshRect(rRight.x, rRight.y, rRight.w, rRight.h, rightA >= 1 ? 1 : 0);
                meshRect(rUp.x, rUp.y, rUp.w, rUp.h, upA >= 1 ? 1 : 0);
                meshRect(rDown.x, rDown.y, rDown.w, rDown.h, downA >= 1 ? 1 : 0);
                meshRect(rCenter.x, rCenter.y, rCenter.w, rCenter.h, centerA >= 1 ? 1 : 0);

                // hint text centered
                if (hint) {
                    ctx2.font = "18px Arial";
                    ctx2.fillStyle = "rgba(255,255,255,0.95)";
                    ctx2.textAlign = "center";
                    ctx2.fillText(hint, x + w / 2.0, Math.max(26, y - 16));
                    ctx2.textAlign = "left";
                }
            }

            return debug;
        }
        """,
        Output("debug-clicks", "children"),
        Input("bbox-store", "data"),  # Используем bbox-store напрямую
        Input("status-mode", "children"),
        Input("status-recognition", "children"),
        Input("fps-label", "children"),
        prevent_initial_call=True,  # True - callback срабатывает только после первого обновления bbox-store от сервера
    )

    # 4) Серверная обработка (register/recognize) — использует ту же систему, что и FastAPI
    @dash_app.callback(
        Output("result", "children"),
        Output("status-recognition", "children"),
        Output("bbox-store", "data"),  # УБРАЛ allow_duplicate - он блокирует обновления
        Output("enroll-store", "data"),
        Output("status-enroll", "children"),
        Output("enroll-progress", "value"),
        Input("frame-store", "data"),
        State("enroll-store", "data"),
        prevent_initial_call=False,  # False - callback срабатывает сразу
    )
    def handle_frame(data: Dict[str, Any], enroll_store: Optional[Dict[str, Any]]):
        # КРИТИЧЕСКОЕ ЛОГИРОВАНИЕ: проверяем, что callback вызывается
        print(f"[handle_frame] CALLBACK FIRED: data={data is not None} action={data.get('action') if data else None} ts={data.get('ts') if data else None}", flush=True)
        
        # ВАЖНО: всегда возвращаем bbox-store (хотя бы с ok=False), чтобы clientside callback видел обновление
        import time
        empty_bbox = {"ok": False, "action": data.get("action") if data else None, "ts": data.get("ts") if data else None, "frame": {"w": 0, "h": 0}, "mesh": {"points": []}, "mesh_n": 0, "_update_id": int(time.time() * 1000)}
        if not data or "frame" not in data:
            print(f"[handle_frame] SKIP: no data or no frame data={data}", flush=True)
            return "", "", empty_bbox, enroll_store, "", 0

        action = data.get("action")
        frame_url = data.get("frame", "")
        now_s = datetime.now().strftime("%H:%M:%S")
        ts = data.get("ts")

        try:
            raw = _decode_data_url_to_bytes(frame_url)
            img_bgr = decode_image_bytes(raw)
        except Exception as e:
            # иначе в UI выглядит как "мертвый прогресс" + нет диагностики
            return "", f"Ошибка кадра: {e}", {"ok": False, "action": action, "ts": ts, "frame": {"w": 0, "h": 0}, "mesh": {"points": []}, "mesh_n": 0, "_update_id": int(time.time() * 1000)}, enroll_store, f"Ракурсы: ошибка кадра • {now_s}", 0

        # Размер кадра нужен даже если лицо не найдено (для корректного маппинга на клиенте)
        ih, iw = img_bgr.shape[:2]

        try:
            info = detect_face_info_fn(img_bgr) or {}
            face_coords = info.get("bbox") or detect_face_fn(img_bgr)
        except ValueError:
            message = "Не удалось однозначно выбрать лицо. Поместите одно лицо в центр кадра."
            empty_bbox["frame"] = {"w": int(iw), "h": int(ih)}
            return message, message, empty_bbox, enroll_store, message, 0
        
        # facemesh landmarks (106 points) в нормализованных координатах кадра (0..1)
        mesh_points = []
        if face_coords is not None:
            x, y, w, h = face_coords
            try:
                lmk106 = info.get("lmk106")
                if isinstance(lmk106, list) and len(lmk106) >= 50:
                    for p in lmk106:
                        if not isinstance(p, (list, tuple)) or len(p) < 2:
                            continue
                        mesh_points.append([float(p[0]) / float(iw), float(p[1]) / float(ih)])
            except Exception:
                mesh_points = []

            # No synthetic landmark grid: without measured landmarks draw bbox only.

        # ТЕПЕРЬ все переменные (iw, ih, mesh_points, action, ts) определены — можно создать closure
        def _bbox_payload(*, ok: bool, bbox: Optional[dict] = None, enroll: Optional[dict] = None) -> Dict[str, Any]:
            import time
            out: Dict[str, Any] = {
                "ok": bool(ok), 
                "frame": {"w": int(iw), "h": int(ih)}, 
                "action": action, 
                "ts": ts, 
                "mesh_n": int(len(mesh_points)),
                "_update_id": int(time.time() * 1000)  # УНИКАЛЬНЫЙ ID для каждого обновления - заставляет Dash обновлять Store
            }
            if bbox is not None:
                out["bbox"] = bbox
            out["mesh"] = {"points": mesh_points}
            if enroll is not None:
                out["enroll"] = enroll
            # КРИТИЧЕСКАЯ ДИАГНОСТИКА: проверяем, что payload действительно создаётся
            try:
                import json
                payload_str = json.dumps(out, default=str)
                payload_size = len(payload_str)
                print(f"[handle_frame] _bbox_payload created: ok={out.get('ok')} mesh_n={out.get('mesh_n')} payload_size={payload_size} keys={list(out.keys())} _update_id={out.get('_update_id')}", flush=True)
            except Exception as e:
                print(f"[handle_frame] _bbox_payload error: {e}", flush=True)
            return out
        
        if face_coords is None:
            # Если 3D трекинг включен, но лицо не найдено — явно сообщаем в UI.
            payload_no_face = _bbox_payload(ok=False, bbox=None, enroll=None)
            try:
                print(f"[handle_frame] NO FACE: action={action} ts={ts} iw={iw} ih={ih} mesh_n={len(mesh_points)}", flush=True)
            except Exception:
                pass
            return (
                "",
                "Лицо не найдено",
                payload_no_face,
                enroll_store,
                f"Ракурсы: лицо не найдено • {now_s} • action={action or '—'}",
                0,
            )

        x, y, w, h = face_coords
        bbox_norm = {
            "x": float(x) / float(iw),
            "y": float(y) / float(ih),
            "w": float(w) / float(iw),
            "h": float(h) / float(ih),
        }
        face_img = img_bgr[y : y + h, x : x + w]
        try:
            print(f"[handle_frame] FACE OK: action={action} ts={ts} bbox={w}x{h} mesh_n={len(mesh_points)} enroll_active={isinstance(enroll_store, dict) and enroll_store.get('active')}", flush=True)
        except Exception:
            pass

        # --- 3D enrollment (FaceID-like): собираем разные позы головы ---
        def pose_bin_from_kps(kps_list):
            # простая (но устойчивая) оценка yaw/pitch по 5 точкам: глаза, нос, рот
            # kps: [Leye, Reye, Nose, Lmouth, Rmouth]
            try:
                kps = np.asarray(kps_list, dtype=np.float32)
                if kps.shape != (5, 2):
                    return "center", 0.0, 0.0
                le, re, nose, lm, rm = kps
                eye_mid = (le + re) / 2.0
                mouth_mid = (lm + rm) / 2.0
                inter = float(np.linalg.norm(re - le) + 1e-6)
                yaw = float((nose[0] - eye_mid[0]) / (0.5 * inter))  # -1..1 примерно
                pitch = float(((nose[1] - eye_mid[1]) / (mouth_mid[1] - eye_mid[1] + 1e-6)) - 0.5) * 2.0
                # биннинг
                if yaw < -0.55:
                    return "left", yaw, pitch
                if yaw > 0.55:
                    return "right", yaw, pitch
                if pitch < -0.55:
                    return "up", yaw, pitch
                if pitch > 0.55:
                    return "down", yaw, pitch
                return "center", yaw, pitch
            except Exception:
                return "center", 0.0, 0.0

        try:
            if action == "track":
                # Always-on 3D tracking: обновляем mesh + bbox + (опционально) overlay зон, но НИЧЕГО не сохраняем.
                overlay = {"hint": "", "zones": {"center": 0.0, "left": 0.0, "right": 0.0, "up": 0.0, "down": 0.0}, "progress": 0.0}
                status = f"Ракурсы: активен • {now_s} • action={action or '—'} • mesh={len(mesh_points)}"
                try:
                    if isinstance(enroll_store, dict) and enroll_store.get("need") and enroll_store.get("got"):
                        need = enroll_store.get("need", {})
                        got = enroll_store.get("got", {})
                        # reuse the same logic as enroll_overlay would compute
                        def _r(k: str) -> float:
                            n = float(need.get(k, 0) or 0)
                            g = float(got.get(k, 0) or 0)
                            if n <= 0:
                                return 0.0
                            return float(max(0.0, min(1.0, g / n)))
                        overlay["zones"] = {"center": _r("center"), "left": _r("left"), "right": _r("right"), "up": _r("up"), "down": _r("down")}
                        total_need = sum(int(v) for v in need.values()) or 1
                        total_got = sum(int(v) for v in got.values())
                        overlay["progress"] = float(max(0.0, min(1.0, total_got / float(total_need))))

                        # Понятный прогресс по зонам (сколько кадров осталось)
                        def _rem(k: str) -> str:
                            try:
                                return f"{int(got.get(k,0))}/{int(need.get(k,0))}"
                            except Exception:
                                return "0/0"
                        rem = 0
                        for k in ["center","left","right","up","down"]:
                            try:
                                rem += max(0, int(need.get(k, 0)) - int(got.get(k, 0)))
                            except Exception:
                                pass
                        # enroll-interval = 240ms => грубая оценка времени, если держать лицо стабильно
                        eta_s = max(0.0, rem * 0.24)
                        status = (
                            f"Ракурсы: активен • {now_s} • "
                            f"центр {_rem('center')}, "
                            f"левая {_rem('left')}, "
                            f"правая {_rem('right')}, "
                            f"лоб {_rem('up')}, "
                            f"подбородок {_rem('down')} • "
                            f"осталось ~{rem} кадров (~{eta_s:.1f}с)"
                        )
                except Exception:
                    pass

                bbox_payload_out = _bbox_payload(ok=True, bbox=bbox_norm, enroll=overlay)
                try:
                    mesh_pts_count = len(bbox_payload_out.get("mesh", {}).get("points", []))
                    payload_keys = list(bbox_payload_out.keys())
                    import json
                    payload_json = json.dumps(bbox_payload_out, default=str)
                    print(f"[handle_frame] TRACK return: ok={bbox_payload_out.get('ok')} mesh_n={mesh_pts_count} prog={overlay.get('progress', 0.0):.2f} bbox_w={bbox_norm.get('w', 0):.3f} keys={payload_keys} payload_size={len(payload_json)}", flush=True)
                    print(f"[handle_frame] TRACK bbox-store будет обновлён: {bbox_payload_out.get('ok')} mesh_n={mesh_pts_count} _update_id={bbox_payload_out.get('_update_id')}", flush=True)
                except Exception as e:
                    print(f"[handle_frame] TRACK logging error: {e}", flush=True)
                # ВАЖНО: возвращаем реальные значения, чтобы Dash обновил bbox-store
                return (
                    "",
                    status,
                    bbox_payload_out,  # ВСЕГДА обновляем bbox-store
                    enroll_store,
                    (f"Ракурсы: запись включена • {now_s} • action={action or '—'} • mesh={len(mesh_points)}" if isinstance(enroll_store, dict) and enroll_store.get("active") else status),
                    int(round(float(overlay.get("progress", 0.0) or 0.0) * 100.0)),
                )

            # Для enroll/recognize система нужна (запись/поиск в базе). Для track — нет.
            system = init_system()
            if system is None:
                sys_not_ready_payload = {"ok": True, "bbox": bbox_norm, "frame": {"w": int(iw), "h": int(ih)}, "mesh": {"points": mesh_points}, "action": action, "ts": ts, "mesh_n": len(mesh_points), "_update_id": int(time.time() * 1000)}
                return (
                    "",
                    "Система распознавания не инициализировалась (проверь логи сервера)",
                    sys_not_ready_payload,
                    enroll_store,
                    "Ракурсы: активен (трек) • система распознавания ещё не готова",
                    0,
                )

            if action == "enroll":
                # UI "зоны" лица (идея как у сканеров): лоб/щеки/подбородок/центр.
                zone_label = {
                    "center": "Центр",
                    "left": "Левая щека",
                    "right": "Правая щёка",
                    "up": "Лоб",
                    "down": "Подбородок",
                }

                def enroll_overlay(hint_key: str, need_d: Dict[str, Any], got_d: Dict[str, Any]) -> Dict[str, Any]:
                    def r(k: str) -> float:
                        try:
                            n = float(need_d.get(k, 0) or 0)
                            g = float(got_d.get(k, 0) or 0)
                            if n <= 0:
                                return 0.0
                            return float(max(0.0, min(1.0, g / n)))
                        except Exception:
                            return 0.0

                    # подсказываем следующую зону (в терминах UI)
                    hint_map = {
                        "center": "Смотри прямо (центр)",
                        "left": "Поверни голову влево (левая щека)",
                        "right": "Поверни голову вправо (правая щёка)",
                        "up": "Чуть вверх (лоб)",
                        "down": "Чуть вниз (подбородок)",
                        "done": "Готово",
                    }
                    zones = {"center": r("center"), "left": r("left"), "right": r("right"), "up": r("up"), "down": r("down")}
                    total_need = sum(int(v) for v in (need_d or {}).values()) or 1
                    total_got = sum(int(v) for v in (got_d or {}).values())
                    progress = float(max(0.0, min(1.0, total_got / float(total_need))))
                    return {"hint": hint_map.get(hint_key, ""), "zones": zones, "progress": progress}

                # init store if missing
                if not enroll_store or not enroll_store.get("active"):
                    enroll_init_payload = _bbox_payload(ok=True, bbox=bbox_norm, enroll={"hint": "", "zones": {}, "progress": 0.0})
                    return (
                        "",
                        "",
                        enroll_init_payload,
                        enroll_store,
                        "Регистрация: нажмите «Запомнить лицо», чтобы начать.",
                        0,
                    )

                kps = info.get("kps")
                kps_array = np.asarray(kps, dtype=np.float32) if kps is not None else np.empty((0, 2))
                pose_available = kps_array.shape == (5, 2) and np.isfinite(kps_array).all()
                if not enroll_store.get("capture_mode"):
                    enroll_store["capture_mode"] = "multi_view" if pose_available else "frontal"
                    if not pose_available:
                        enroll_store["need"] = {"center": 3}
                        enroll_store["got"] = {"center": 0}
                if enroll_store["capture_mode"] == "frontal":
                    bin_name, yaw, pitch = "center", None, None
                elif not pose_available:
                    message = "Ориентиры лица временно не определены; смотрите в камеру. Кадр не сохранён."
                    return "", message, _bbox_payload(ok=True, bbox=bbox_norm), enroll_store, message, 0
                else:
                    bin_name, yaw, pitch = pose_bin_from_kps(kps)
                need = enroll_store.get("need", {})
                got = enroll_store.get("got", {})
                if bin_name not in need:
                    bin_name = "center"

                # если уже набрано — ищем следующий недобор
                if got.get(bin_name, 0) >= need.get(bin_name, 0):
                    # подсказка: куда повернуть дальше
                    next_targets = [k for k in ["center", "left", "right", "up", "down"] if got.get(k, 0) < need.get(k, 0)]
                    hint = next_targets[0] if next_targets else "done"
                    total_need = sum(int(v) for v in need.values())
                    total_got = sum(int(v) for v in got.values())
                    prog = int(round(100 * total_got / max(total_need, 1)))
                    enroll_next_payload = _bbox_payload(ok=True, bbox=bbox_norm, enroll=enroll_overlay(hint, need, got))
                    return (
                        "",
                        "",
                        enroll_next_payload,
                        enroll_store,
                        f"Ракурсы: следующая зона — {zone_label.get(hint,'')}. Прогресс {prog}%.",
                        prog,
                    )

                # добавляем эмбеддинг для этого бина
                name = str(enroll_store.get("name") or "").strip()
                if not name:
                    enroll_name_payload = _bbox_payload(ok=True, bbox=bbox_norm, enroll=enroll_overlay("center", need, got))
                    return (
                        "",
                        "Введите имя",
                        enroll_name_payload,
                        enroll_store,
                        "Введите имя.",
                        0,
                    )
                person_id = _safe_person_id_from_name(name)
                system.register_person(
                    image=img_bgr if getattr(system, "input_mode", "face_crop") == "full_frame" else face_img,
                    person_id=person_id,
                    person_name=name,
                    additional_metadata={"source": "dash", "capture_mode": enroll_store["capture_mode"],
                                         "pose_bin": bin_name, "yaw_proxy": yaw, "pitch_proxy": pitch},
                )
                got[bin_name] = int(got.get(bin_name, 0)) + 1
                enroll_store["got"] = got

                total_need = sum(int(v) for v in need.values())
                total_got = sum(int(v) for v in got.values())
                prog = int(round(100 * total_got / max(total_need, 1)))

                next_targets = [k for k in ["center", "left", "right", "up", "down"] if got.get(k, 0) < need.get(k, 0)]
                hint = next_targets[0] if next_targets else "done"
                if hint == "done":
                    enroll_store["active"] = False
                
                enroll_ov = enroll_overlay(hint, need, got)
                try:
                    print(f"[handle_frame] ENROLL return: ok=True mesh_n={len(mesh_points)} prog={prog}% bin={bin_name} hint={hint} got={got} bbox_w={bbox_norm.get('w', 0):.3f}", flush=True)
                except Exception:
                    pass

                enroll_saved_payload = _bbox_payload(ok=True, bbox=bbox_norm, enroll=enroll_ov)
                mode_label = "Фронтальная регистрация" if enroll_store["capture_mode"] == "frontal" else "Ракурсы"
                return (
                    f"Ракурсы: сохранено — {zone_label.get(bin_name, bin_name)} ({got[bin_name]}/{need[bin_name]})",
                    "",
                    enroll_saved_payload,
                    enroll_store,
                    (f"{mode_label}: завершено ✅" if hint == "done" else
                     f"{mode_label}: смотрите прямо, сохранено {total_got}/{total_need}." if enroll_store["capture_mode"] == "frontal" else
                     f"Ракурсы: следующая зона — {zone_label.get(hint,'')}. Прогресс {prog}%."),
                    prog,
                )

            if action == "recognize":
                recognition_image = img_bgr if getattr(system, "input_mode", "face_crop") == "full_frame" else face_img
                rec = system.recognize(recognition_image, k=5, return_distances=True)
                person_name, confidence, candidates, distances = rec

                if not person_name:
                    dbg = []
                    if confidence is not None:
                        dbg.append(f"сходство: {confidence:.3f}; не вероятность")
                    dbg_s = (" (" + ", ".join(dbg) + ")") if dbg else ""
                    rec_not_payload = _bbox_payload(ok=True, bbox=bbox_norm, enroll=None)
                    return (
                        f"Не распознано{dbg_s}",
                        f"Последний результат: не распознано{dbg_s}",
                        rec_not_payload,
                        enroll_store,
                        "",
                        0,
                    )
                rec_ok_payload = _bbox_payload(ok=True, bbox=bbox_norm, enroll=None)
                return (
                    f"Распознан: {person_name} (сходство: {confidence:.3f}; не вероятность)",
                    f"Последний результат: {person_name} • сходство: {confidence:.3f}",
                    rec_ok_payload,
                    enroll_store,
                    "",
                    0,
                )

            return "", "Неизвестное действие", {"ok": False, "action": action, "ts": ts, "frame": {"w": 0, "h": 0}, "mesh": {"points": []}, "mesh_n": 0, "_update_id": int(time.time() * 1000)}, enroll_store, "", 0
        except Exception as e:
            err_payload = _bbox_payload(ok=True, bbox=bbox_norm, enroll=None)
            return (
                f"Ошибка: {e}",
                f"Ошибка: {e}",
                err_payload,
                enroll_store,
                f"Ошибка: {e}",
                0,
            )

    @dash_app.callback(
        Output("debug-clicks", "title"),
        Input("btn-start", "n_clicks"),
        Input("btn-stop", "n_clicks"),
        Input("btn-enroll-3d", "n_clicks"),
        Input("btn-recognize-toggle", "n_clicks"),
        prevent_initial_call=False,
    )
    def _debug_clicks(n_start, n_stop, n_enroll, n_toggle):
        return f"debug: start={n_start or 0}, stop={n_stop or 0}, enroll={n_enroll or 0}, toggle={n_toggle or 0}"

    return dash_app


