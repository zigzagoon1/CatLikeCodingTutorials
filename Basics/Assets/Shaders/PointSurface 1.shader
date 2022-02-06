Shader "Custom/PointSurface"
{
    Properties
    {
        _Smoothness("Smoothness", Range(0,1)) = 0.5
    }
    SubShader
    {
        CGPROGRAM
        
        #pragma surface ConfigureSurface Standard fullforwardshadows
        #pragma target 3.0

        float _Smoothness;

        struct Input
        {
            float3 worldPos;
        };

        void ConfigureSurface(Input input, inout SurfaceOutputStandard surface)
        {
            surface.Albedo = saturate(input.worldPos * 0.5 + 0.5);
            surface.Smoothness = _Smoothness;
        }

        ENDCG
    }
        

    
    FallBack "Diffuse"
}
