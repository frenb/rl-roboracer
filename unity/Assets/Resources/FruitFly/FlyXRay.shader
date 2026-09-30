// See-through shell for the fly anatomy view. Unlit, so it needs no scene
// lights on the overlay layer. Opacity is a floor across the whole surface
// plus a fresnel rim, which keeps the silhouette readable at a floor low
// enough to see the connectome through.
//
// Lives in Resources so it ships in a build: FlyAnatomyView finds it with
// Shader.Find and nothing else references it.
Shader "Hidden/FlyXRay"
{
    Properties
    {
        _MainTex ("Albedo", 2D) = "white" {}
        _Color ("Tint", Color) = (0.75, 0.85, 1.0, 1)
        _TexStrength ("Albedo strength", Range(0, 1)) = 0.35
        _Opacity ("Opacity", Range(0, 1)) = 0.1
        _RimColor ("Rim colour", Color) = (0.6, 0.85, 1.0, 1)
        _RimOpacity ("Rim opacity", Range(0, 1)) = 0.45
        _RimPower ("Rim power", Range(0.5, 8)) = 2.5
        _RearFade ("Opacity scale where uv2.x is 0", Range(0, 1)) = 1
        _MouthFade ("Opacity scale where uv2.y is 1", Range(0, 1)) = 1
        _HeadFade ("Opacity scale where uv2.z is 1", Range(0, 1)) = 1
    }
    SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType" = "Transparent" "IgnoreProjector" = "True" }
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Back
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_ST;
            fixed4 _Color, _RimColor;
            half _TexStrength, _Opacity, _RimOpacity, _RimPower, _RearFade, _MouthFade, _HeadFade;

            // uv2.x (mesh UV channel 2) is 1 on the parts _RearFade leaves
            // alone - head, front legs, wing roots - and 0 elsewhere; uv2.y is
            // 1 on the mouthparts, which _MouthFade scales; uv2.z is 1 on the
            // head, which _HeadFade scales. All are written from the skin
            // weights by FlyAnatomyView. A mesh without the channel reads 0.
            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float2 uv : TEXCOORD0;
                float3 keep : TEXCOORD2;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 normal : TEXCOORD1;
                float3 viewDir : TEXCOORD2;
                float3 keep : TEXCOORD3;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                o.keep = saturate(v.keep);
                o.normal = UnityObjectToWorldNormal(v.normal);
                // Per-vertex view direction, not the camera forward: correct
                // under the orthographic overlay camera and a perspective one.
                o.viewDir = UnityWorldSpaceViewDir(mul(unity_ObjectToWorld, v.vertex).xyz);
                if (unity_OrthoParams.w > 0.5)
                    o.viewDir = -UNITY_MATRIX_V[2].xyz;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float3 n = normalize(i.normal);
                float3 v = normalize(i.viewDir);
                half rim = pow(1.0 - saturate(abs(dot(n, v))), _RimPower);
                fixed3 tex = tex2D(_MainTex, i.uv).rgb;
                fixed3 body = _Color.rgb * lerp(fixed3(1, 1, 1), tex, _TexStrength);
                fixed3 col = lerp(body, _RimColor.rgb, rim);
                half a = saturate(_Opacity + _RimOpacity * rim) * _Color.a
                         * lerp(_RearFade, 1.0, i.keep.x) * lerp(1.0, _MouthFade, i.keep.y) * lerp(1.0, _HeadFade, i.keep.z);
                return fixed4(col, a);
            }
            ENDCG
        }
    }
}
