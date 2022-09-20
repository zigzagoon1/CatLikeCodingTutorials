using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

using static Unity.Mathematics.math;
public abstract class Visualization : MonoBehaviour
{
    public enum Shape { Plane, Sphere, Torus }

    static Shapes.ScheduleDelegate[] shapeJobs = {
        Shapes.Job<Shapes.Plane>.ScheduleParallel,
        Shapes.Job<Shapes.Sphere>.ScheduleParallel,
        Shapes.Job<Shapes.Torus>.ScheduleParallel
    };

    static int
        positionsID = Shader.PropertyToID("_Positions"),
        normalsID = Shader.PropertyToID("_Normals"),
        configID = Shader.PropertyToID("_Config");

    [SerializeField]
    Mesh instanceMesh;

    [SerializeField]
    Material material;

    [SerializeField, Range(0.1f, 10f)]
    float instanceScale = 2f;

    [SerializeField, Range(1, 512)]
    int resolution = 16;

    [SerializeField, Range(-0.5f, 0.5f)]
    float displacement = 0.1f;

    [SerializeField]
    Shape shape;

    NativeArray<float3x4> positions, normals;

    ComputeBuffer positionsBuffer, normalsBuffer;

    MaterialPropertyBlock propertyBlock;

    bool isDirty;

    Bounds bounds;


    private void OnEnable()
    {
        isDirty = true;

        int length = resolution * resolution;
        length = length / 4 + (length & 1);
        positions = new NativeArray<float3x4>(length, Allocator.Persistent);
        normals = new NativeArray<float3x4>(length, Allocator.Persistent);
        positionsBuffer = new ComputeBuffer(length * 4, 3 * 4);
        normalsBuffer = new ComputeBuffer(length * 4, 3 * 4);

        propertyBlock ??= new MaterialPropertyBlock();
        EnableVisualization(length, propertyBlock);
        propertyBlock.SetBuffer(positionsID, positionsBuffer);
        propertyBlock.SetBuffer(normalsID, normalsBuffer);
        propertyBlock.SetVector(configID, new Vector4(resolution, instanceScale / resolution, displacement));
    }

    private void OnDisable()
    {
        positions.Dispose();
        normals.Dispose();

        positionsBuffer.Release();
        normalsBuffer.Release();

        positionsBuffer = null;
        normalsBuffer = null;

        DisableVisualization();
    }

    private void OnValidate()
    {
        if (positionsBuffer != null && enabled)
        {
            OnDisable();
            OnEnable();
        }
    }

    private void Update()
    {
        if (isDirty || transform.hasChanged)
        {
            isDirty = false;
            transform.hasChanged = false;

            UpdateVisualization(positions, resolution, shapeJobs[(int)shape](positions, normals, resolution, transform.localToWorldMatrix, default));

            positionsBuffer.SetData(positions.Reinterpret<float3>(3 * 4 * 4));
            normalsBuffer.SetData(normals.Reinterpret<float3>(3 * 4 * 4));

            bounds = new Bounds(transform.position, float3(2f * cmax(abs(transform.lossyScale)) + displacement));

        }
        Graphics.DrawMeshInstancedProcedural(instanceMesh, 0, material, bounds, resolution * resolution, propertyBlock);
    }

    protected abstract void EnableVisualization(int dataLength, MaterialPropertyBlock propertyBlock);
    protected abstract void DisableVisualization();

    protected abstract void UpdateVisualization(NativeArray<float3x4> positions, int resolution, JobHandle handle);
}
