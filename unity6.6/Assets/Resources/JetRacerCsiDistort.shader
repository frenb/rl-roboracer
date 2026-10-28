Shader "Hidden/JetRacerCsiDistort"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _K;    // fx, fy, cx, cy  (OpenCV, pixels)
            float4 _D;    // k1, k2, p1, p2
            float _K3;
            float2 _Size; // 640, 480

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            // Output is CSI-like (distorted). Sample the pinhole render at
            // the undistorted pixel — OpenCV iterative invert of plumb_bob.
            fixed4 frag(v2f i) : SV_Target
            {
                float fx = _K.x, fy = _K.y, cx = _K.z, cy = _K.w;
                float k1 = _D.x, k2 = _D.y, p1 = _D.z, p2 = _D.w, k3 = _K3;

                // Unity UV origin is bottom-left; OpenCV is top-left.
                float u = i.uv.x * _Size.x;
                float v = (1.0 - i.uv.y) * _Size.y;

                float xn = (u - cx) / fx;
                float yn = (v - cy) / fy;
                float x = xn;
                float y = yn;
                [unroll]
                for (int n = 0; n < 5; n++)
                {
                    float r2 = x * x + y * y;
                    float r4 = r2 * r2;
                    float icdist = 1.0 / (1.0 + k1 * r2 + k2 * r4 + k3 * r4 * r2);
                    float dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
                    float dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
                    x = (xn - dx) * icdist;
                    y = (yn - dy) * icdist;
                }

                float us = fx * x + cx;
                float vs = fy * y + cy;
                float2 uv = float2(us / _Size.x, 1.0 - vs / _Size.y);
                if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
                    return fixed4(0, 0, 0, 1);
                return tex2D(_MainTex, uv);
            }
            ENDCG
        }
    }
}
