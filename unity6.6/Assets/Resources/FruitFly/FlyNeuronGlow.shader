// Neurons in the brain close-up: each quad is a soft round spot, added to what
// is behind it rather than blended over it, so dense firing regions bloom and a
// quiet brain still reads as a lit shape against the dark head. Colour and
// weight come from the vertex colour (rgb, a) that FlyBrainViz writes from
// activity.
//
// In Resources so it ships in a build; FlyBrainViz finds it with Shader.Find.
Shader "Hidden/FlyNeuronGlow"
{
    Properties
    {
        _Gain ("Gain", Range(0, 4)) = 1
        _Core ("Core sharpness", Range(0.5, 8)) = 2
    }
    SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType" = "Transparent" "IgnoreProjector" = "True" }
        Blend One One
        ZWrite Off
        ZTest Always
        Cull Off
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            half _Gain, _Core;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                fixed4 color : COLOR;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
                fixed4 color : COLOR;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv * 2.0 - 1.0;
                o.color = v.color;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                half f = pow(saturate(1.0 - length(i.uv)), _Core);
                return fixed4(i.color.rgb * (i.color.a * f * _Gain), 0);
            }
            ENDCG
        }
    }
}
