using System.Collections;
using System.IO;
using RosMessageTypes.Std;
using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// Phase 1 of docs/csi-camera-observation-guide.md: one downsampled CSI
/// frame per <c>cmd_id</c> on <c>camera/front</c>. Not a 20 Hz stream.
/// Uses a dedicated publish RT so P-view undocking the display RT does
/// not stop capture. Auto-attached by SimController.
///
/// Success check (Unity-local; ros-server drops this topic until Phase 2):
/// console <c>[CsiFramePublisher] ok</c>, HUD line, PNGs in
/// <c>unity/CsiFrameDumps/</c> (auto first 3, or press F).
/// </summary>
public class CsiFramePublisher : MonoBehaviour
{
    public const string Topic = "camera/front";
    public const string CameraName = "JetRacerCsiCamera";
    public const string FrameId = "camera_visual";
    public const int SrcWidth = JetRacerCsiIntrinsics.Width;
    public const int SrcHeight = JetRacerCsiIntrinsics.Height;
    public const int PubWidth = 84;
    public const int PubHeight = 84;

    public static CsiFramePublisher Instance { get; private set; }

    public static int PublishCount { get; private set; }
    public static int LastCmdId { get; private set; }
    public static int LastBytes { get; private set; }
    public static string LastDumpPath { get; private set; }
    public static string LastStatus { get; private set; } = "csi: waiting";

    ROSConnection _ros;
    RenderTexture _fullRt;
    RenderTexture _smallRt;
    Texture2D _readTex;
    bool _capturing;
    int _pendingCmdId = int.MinValue;
    int _lastPublishedCmdId = int.MinValue;
    int _autoDumpsLeft = 3;

    void Awake()
    {
        Instance = this;
        EnsureBuffers();
        LastStatus = "csi: ready";
    }

    void OnEnable()
    {
        Instance = this;
        EnsureBuffers();
    }

    void OnDisable()
    {
        if (Instance == this) Instance = null;
    }

    void Start()
    {
        _ros = ROSConnection.instance;
    }

    void EnsureBuffers()
    {
        if (_fullRt == null) _fullRt = NewRt(SrcWidth, SrcHeight, 16);
        if (_smallRt == null) _smallRt = NewRt(PubWidth, PubHeight, 0);
        if (_readTex == null)
            _readTex = new Texture2D(PubWidth, PubHeight, TextureFormat.RGB24, false);
    }

    void OnDestroy()
    {
        if (_fullRt != null) _fullRt.Release();
        if (_smallRt != null) _smallRt.Release();
        if (_readTex != null) Destroy(_readTex);
    }

    void Update()
    {
        var kb = Keyboard.current;
        if (kb != null && kb.fKey.wasPressedThisFrame)
            Request(_lastPublishedCmdId == int.MinValue ? 0 : _lastPublishedCmdId, forceDump: true);
    }

    /// <summary>One capture per new cmd_id (ApplyForce / reset).</summary>
    public void Request(int cmdId, bool forceDump = false)
    {
        if (!forceDump && cmdId == _lastPublishedCmdId && cmdId == _pendingCmdId)
            return;
        _pendingCmdId = cmdId;
        if (_capturing) return;
        StartCoroutine(Capture(cmdId, forceDump));
    }

    IEnumerator Capture(int cmdId, bool forceDump)
    {
        _capturing = true;
        EnsureBuffers();
        if (_ros == null) _ros = ROSConnection.instance;
        yield return new WaitForEndOfFrame();

        var cam = FindCsiCamera();
        if (cam == null)
        {
            LastStatus = "csi: no JetRacerCsiCamera";
            Debug.LogWarning("[CsiFramePublisher] no " + CameraName +
                             " (car not spawned yet?) cmd_id=" + cmdId);
            _capturing = false;
            yield break;
        }

        var savedTarget = cam.targetTexture;
        bool savedEnabled = cam.enabled;
        cam.targetTexture = _fullRt;
        cam.enabled = true;
        cam.Render();
        Graphics.Blit(_fullRt, _smallRt);
        cam.targetTexture = savedTarget;
        cam.enabled = savedEnabled;

        var prev = RenderTexture.active;
        RenderTexture.active = _smallRt;
        _readTex.ReadPixels(new Rect(0, 0, PubWidth, PubHeight), 0, 0, false);
        _readTex.Apply(false, false);
        RenderTexture.active = prev;

        // EncodePNG expects Unity's bottom-left Texture2D. ROS Image (0,0)
        // is top-left, so only the published rgb8 bytes are flipped.
        var px = _readTex.GetPixels32();
        bool dump = forceDump || _autoDumpsLeft > 0;
        if (dump)
        {
            if (!forceDump) _autoDumpsLeft--;
            LastDumpPath = WritePng(_readTex, cmdId);
        }

        FlipVert(px, PubWidth, PubHeight);
        byte[] rgb = new byte[PubWidth * PubHeight * 3];
        int o = 0;
        for (int i = 0; i < px.Length; i++)
        {
            rgb[o++] = px[i].r;
            rgb[o++] = px[i].g;
            rgb[o++] = px[i].b;
        }

        var img = new RosMessageTypes.Sensor.Image
        {
            header = new Header
            {
                seq = cmdId < 0 ? 0u : (uint)cmdId,
                stamp = new RosMessageTypes.Std.Time(),
                frame_id = FrameId
            },
            height = (uint)PubHeight,
            width = (uint)PubWidth,
            encoding = "rgb8",
            is_bigendian = 0,
            step = (uint)(PubWidth * 3),
            data = rgb
        };
        var msg = new RosMessageTypes.NiryoMoveit.Camera(img);

        if (_ros != null)
            _ros.Send(Topic, msg);

        PublishCount++;
        LastCmdId = cmdId;
        LastBytes = rgb.Length;
        _lastPublishedCmdId = cmdId;

        LastStatus = "csi cmd " + cmdId + " " + PubWidth + "x" + PubHeight +
                     " #" + PublishCount;
        Debug.Log("[CsiFramePublisher] ok topic=" + Topic +
                  " cmd_id=" + cmdId +
                  " " + PubWidth + "x" + PubHeight +
                  " encoding=rgb8 bytes=" + rgb.Length +
                  " seq=" + img.header.seq +
                  (dump ? " dump=" + LastDumpPath : "") +
                  " (ros-server ignores this topic until Phase 2)");

        _capturing = false;
        if (_pendingCmdId != cmdId)
            StartCoroutine(Capture(_pendingCmdId, false));
    }

    static Camera FindCsiCamera()
    {
        var go = GameObject.Find(CameraName);
        if (go != null) return go.GetComponent<Camera>();
        var sim = SimController.instance;
        if (sim != null && sim.car != null)
        {
            var cams = sim.car.GetComponentsInChildren<Camera>(true);
            for (int i = 0; i < cams.Length; i++)
            {
                if (cams[i] != null && cams[i].gameObject.name == CameraName)
                    return cams[i];
            }
        }
        return null;
    }

    static RenderTexture NewRt(int w, int h, int depth)
    {
        var rt = new RenderTexture(w, h, depth, RenderTextureFormat.ARGB32);
        rt.antiAliasing = 1;
        rt.filterMode = FilterMode.Bilinear;
        rt.Create();
        return rt;
    }

    static void FlipVert(Color32[] px, int w, int h)
    {
        for (int y = 0; y < h / 2; y++)
        {
            int a = y * w;
            int b = (h - 1 - y) * w;
            for (int x = 0; x < w; x++)
            {
                var t = px[a + x];
                px[a + x] = px[b + x];
                px[b + x] = t;
            }
        }
    }

    static string DumpDir()
    {
        if (Application.isEditor)
            return Path.GetFullPath(Path.Combine(Application.dataPath, "..", "CsiFrameDumps"));
        return Path.Combine(Application.persistentDataPath, "CsiFrameDumps");
    }

    static string WritePng(Texture2D tex, int cmdId)
    {
        string dir = DumpDir();
        Directory.CreateDirectory(dir);
        string name = "csi_cmd" + cmdId + "_" +
                      System.DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".png";
        string path = Path.Combine(dir, name);
        File.WriteAllBytes(path, tex.EncodeToPNG());
        return path;
    }
}
