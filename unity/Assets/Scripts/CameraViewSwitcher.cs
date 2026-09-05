using System.Collections.Generic;
using UnityEngine;
// Active Input Handling is "Input System Package (New)" only, matching
// HudOverlay's toggle-key pattern.
using UnityEngine.InputSystem;

/// <summary>
/// Toggles the Game view between the scene's top-down Main Camera and the
/// on-stalk <c>JetRacerCsiCamera</c>. Press P to switch. Auto-attached by
/// SimController like HudOverlay — no scene setup required.
///
/// The prefab <c>JetRacerCsiCamera</c> is snapped just in front of the
/// visual camera-stalk lens. P-view undocks the RT so the CSI camera
/// renders to Display 1 (Game view needs a real camera), letterboxed
/// 4:3, with yaml K / D from JetRacerCsiIntrinsics. RT / rect restore
/// when you leave P.
///
/// JetRacerCsiCamera defaults to a clean CSI-like view: no trajectory fan,
/// no goal gates, no perception rays. Those still render on the overhead
/// Main Camera. While the car camera is up, T / G / R toggle those overlays
/// on the CSI view only (overhead T / G / R state is left alone).
/// </summary>
public class CameraViewSwitcher : MonoBehaviour
{
    const string OverheadCameraName = "Main Camera";
    const string CarCameraName = "JetRacerCsiCamera";
    const string GoalMarkerName = "GoalMarkerSphere";
    const string SkyDomeName = "Sky_Dome";

    public static bool CarCameraOn { get; private set; }

    bool _carCameraOn = false;
    // CSI view starts with all debug overlays off.
    bool _csiShowsGoals = false;
    bool _csiShowsTrajectory = false;
    bool _csiShowsRays = false;
    Camera _overhead;
    Camera _carCam;
    Camera _letterboxClear;
    RenderTexture _savedCarTarget;
    Rect _savedCarRect;
    Camera _savedFor;
    bool _haveSavedState;
    readonly List<Renderer> _hiddenForCsi = new List<Renderer>(128);
    readonly List<Renderer> _overlayScratch = new List<Renderer>(128);
    float _nextOverlayRefresh;

    void OnEnable()
    {
        Camera.onPreCull += OnAnyCameraPreCull;
        Camera.onPostRender += OnAnyCameraPostRender;
    }

    void OnDisable()
    {
        Camera.onPreCull -= OnAnyCameraPreCull;
        Camera.onPostRender -= OnAnyCameraPostRender;
        RestoreHiddenRenderers();
        RestoreCsiDisplayState();
        if (_letterboxClear != null) _letterboxClear.enabled = false;
        CarCameraOn = false;
    }

    void Update()
    {
        var kb = Keyboard.current;
        if (kb != null && kb.pKey.wasPressedThisFrame)
        {
            _carCameraOn = !_carCameraOn;
            CarCameraOn = _carCameraOn;
            if (_carCameraOn)
            {
                _csiShowsGoals = false;
                _csiShowsTrajectory = false;
                _csiShowsRays = false;
            }
            Debug.Log("[CameraViewSwitcher] view " +
                      (_carCameraOn ? "JetRacerCsiCamera" : "Main Camera"));
        }
        if (_carCameraOn && kb != null)
        {
            if (kb.tKey.wasPressedThisFrame)
            {
                _csiShowsTrajectory = !_csiShowsTrajectory;
                Debug.Log("[CameraViewSwitcher] CSI trajectory " +
                          (_csiShowsTrajectory ? "ON" : "OFF"));
            }
            if (kb.gKey.wasPressedThisFrame)
            {
                _csiShowsGoals = !_csiShowsGoals;
                Debug.Log("[CameraViewSwitcher] CSI goals " +
                          (_csiShowsGoals ? "ON" : "OFF"));
            }
            if (kb.rKey.wasPressedThisFrame)
            {
                _csiShowsRays = !_csiShowsRays;
                Debug.Log("[CameraViewSwitcher] CSI rays " +
                          (_csiShowsRays ? "ON" : "OFF"));
            }
        }
        Apply();
    }

    void Apply()
    {
        ResolveCameras();
        EnsureCsiIntrinsics();
        if (_carCameraOn)
        {
            if (_carCam == null)
            {
                // Car not spawned yet (or just destroyed on reset) — keep
                // the overhead view so the window does not go black.
                RestoreCsiDisplayState();
                if (_letterboxClear != null) _letterboxClear.enabled = false;
                if (_overhead != null) _overhead.enabled = true;
                return;
            }
            if (_overhead != null) _overhead.enabled = false;
            SaveCsiDisplayStateIfNeeded();
            // Game view only shows cameras that render to a display.
            // The CSI cam's RT must come off while P is up.
            _carCam.targetTexture = null;
            _carCam.targetDisplay = 0;
            _carCam.rect = ViewportLetterbox(4f / 3f);
            _carCam.enabled = true;
            EnsureLetterboxClear().enabled = true;
        }
        else
        {
            RestoreCsiDisplayState();
            if (_letterboxClear != null) _letterboxClear.enabled = false;
            if (_overhead != null) _overhead.enabled = true;
        }
    }

    void SaveCsiDisplayStateIfNeeded()
    {
        if (_carCam == null) return;
        if (_haveSavedState && _savedFor == _carCam) return;
        _savedCarTarget = _carCam.targetTexture as RenderTexture;
        _savedCarRect = _carCam.rect;
        _savedFor = _carCam;
        _haveSavedState = true;
    }

    void RestoreCsiDisplayState()
    {
        if (_haveSavedState && _savedFor != null)
        {
            _savedFor.targetTexture = _savedCarTarget;
            _savedFor.rect = _savedCarRect;
        }
        _haveSavedState = false;
        _savedFor = null;
        _savedCarTarget = null;
    }

    static Rect ViewportLetterbox(float targetAspect)
    {
        float w = Screen.width;
        float h = Screen.height;
        if (w < 1f || h < 1f) return new Rect(0, 0, 1, 1);
        float windowAspect = w / h;
        if (windowAspect > targetAspect)
        {
            float rw = targetAspect / windowAspect;
            return new Rect((1f - rw) * 0.5f, 0f, rw, 1f);
        }
        float rh = windowAspect / targetAspect;
        return new Rect(0f, (1f - rh) * 0.5f, 1f, rh);
    }

    Camera EnsureLetterboxClear()
    {
        if (_letterboxClear != null) return _letterboxClear;
        var go = new GameObject("JetRacerCsiLetterboxClear");
        go.hideFlags = HideFlags.HideAndDontSave;
        _letterboxClear = go.AddComponent<Camera>();
        _letterboxClear.clearFlags = CameraClearFlags.SolidColor;
        _letterboxClear.backgroundColor = Color.black;
        _letterboxClear.cullingMask = 0;
        _letterboxClear.depth = -2;
        _letterboxClear.orthographic = true;
        _letterboxClear.allowHDR = false;
        _letterboxClear.allowMSAA = false;
        _letterboxClear.targetDisplay = 0;
        return _letterboxClear;
    }

    void EnsureCsiIntrinsics()
    {
        if (_carCam == null) return;
        if (_carCam.GetComponent<JetRacerCsiIntrinsics>() == null)
            _carCam.gameObject.AddComponent<JetRacerCsiIntrinsics>();
    }

    void ResolveCameras()
    {
        // Camera.main only returns an *enabled* MainCamera-tagged camera, so
        // after we disable the overhead Camera component it would go null.
        // Find by name (GameObject stays active) and cache it.
        if (_overhead == null)
        {
            var go = GameObject.Find(OverheadCameraName);
            if (go != null) _overhead = go.GetComponent<Camera>();
        }

        // The car is destroyed + re-instantiated every episode; Unity
        // fake-nulls the old Camera, so this re-finds the new one.
        if (_carCam == null)
            _carCam = FindCarCamera();
    }

    static Camera FindCarCamera()
    {
        var go = GameObject.Find(CarCameraName);
        if (go != null) return go.GetComponent<Camera>();

        var sim = SimController.instance;
        if (sim != null && sim.car != null)
        {
            var cams = sim.car.GetComponentsInChildren<Camera>(true);
            for (int i = 0; i < cams.Length; i++)
            {
                if (cams[i] != null && cams[i].gameObject.name == CarCameraName)
                    return cams[i];
            }
        }
        return null;
    }

    bool IsCsiCamera(Camera cam)
    {
        return cam != null && (cam == _carCam || cam.gameObject.name == CarCameraName);
    }

    void OnAnyCameraPreCull(Camera cam)
    {
        if (!IsCsiCamera(cam)) return;
        HideCsiOverlays();
    }

    void OnAnyCameraPostRender(Camera cam)
    {
        if (!IsCsiCamera(cam)) return;
        RestoreHiddenRenderers();
    }

    void HideCsiOverlays()
    {
        CollectHiddenOverlays(_hiddenForCsi);
        for (int i = 0; i < _hiddenForCsi.Count; i++)
        {
            var r = _hiddenForCsi[i];
            if (r != null) r.forceRenderingOff = true;
        }
    }

    void RestoreHiddenRenderers()
    {
        for (int i = 0; i < _hiddenForCsi.Count; i++)
        {
            var r = _hiddenForCsi[i];
            if (r != null) r.forceRenderingOff = false;
        }
        _hiddenForCsi.Clear();
    }

    void CollectHiddenOverlays(List<Renderer> dst)
    {
        dst.Clear();
        if (!_csiShowsGoals)
            AppendGoalRenderers(dst);
        AppendNamedRenderers(dst, SkyDomeName);
        if (!_csiShowsTrajectory)
        {
            var viz = FindObjectOfType<TrajectoryRolloutViz>();
            if (viz != null) viz.CopyLineRenderers(dst);
        }
        if (!_csiShowsRays)
        {
            var car = FindObjectOfType<CarController>();
            if (car != null) car.CopyRayLineRenderers(dst);
        }
    }

    void AppendGoalRenderers(List<Renderer> dst)
    {
        if (Time.unscaledTime < _nextOverlayRefresh && _overlayScratch.Count > 0)
        {
            dst.AddRange(_overlayScratch);
            return;
        }
        _nextOverlayRefresh = Time.unscaledTime + 0.5f;
        _overlayScratch.Clear();
        var goals = FindObjectsOfType<Goal>();
        for (int i = 0; i < goals.Length; i++)
        {
            if (goals[i] == null) continue;
            var rs = goals[i].GetComponentsInChildren<Renderer>(true);
            for (int j = 0; j < rs.Length; j++)
                if (rs[j] != null) _overlayScratch.Add(rs[j]);
        }
        var marker = GameObject.Find(GoalMarkerName);
        if (marker != null)
        {
            var rs = marker.GetComponentsInChildren<Renderer>(true);
            for (int j = 0; j < rs.Length; j++)
                if (rs[j] != null) _overlayScratch.Add(rs[j]);
        }
        dst.AddRange(_overlayScratch);
    }

    static void AppendNamedRenderers(List<Renderer> dst, string name)
    {
        var go = GameObject.Find(name);
        if (go == null) return;
        var rs = go.GetComponentsInChildren<Renderer>(true);
        for (int i = 0; i < rs.Length; i++)
            if (rs[i] != null) dst.Add(rs[i]);
    }
}
