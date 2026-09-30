// Soft dark oval drawn inside the head in the brain close-up, after the fly's
// shell and before the neurons, so the body behind the brain stops showing
// through it. Opaque at the centre, fading to nothing at the quad's edge.
//
// In Resources so it ships in a build; FlyBrainViz finds it with Shader.Find.
Shader "Hidden/FlyBrainBackdrop"
{
    Properties
    {
        _Color ("Colour", Color) = (0, 0, 0, 0.85)
        _Inner ("Solid out to", Range(0, 1)) = 0.6
    }
    SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType" = "Transparent" "IgnoreProjector" = "True" }
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        ZTest Always
        Cull Off
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            fixed4 _Color;
            half _Inner;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv * 2.0 - 1.0;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                half d = length(i.uv);
                return fixed4(_Color.rgb, _Color.a * (1.0 - smoothstep(_Inner, 1.0, d)));
            }
            ENDCG
        }
    }
}
