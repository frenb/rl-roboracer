using System;
using System.Globalization;
using UnityEngine;

/// <summary>
/// Fixed simulated time per trainer action. Once a trainer sends its first
/// sim command, SimController runs exactly <see cref="Ticks"/> physics ticks
/// for each APPLY_FORCE, freezes, and only then publishes the step's
/// observation. Without it the car keeps driving on the last command for as
/// long as the trainer takes to send the next one (a learner update can be
/// several seconds of sim time), so training steps last far longer than eval
/// or demo steps.
///
/// The default matches the ~0.4 s per step the free-running gyms gave eval
/// and demo collection, so existing demos stay on-distribution. Launch with
/// <c>-stepSeconds 0.1</c> to change it, or <c>-stepSeconds 0</c> for the old
/// free-running behaviour.
/// </summary>
public static class Lockstep
{
    public const float DefaultStepSeconds = 0.4f;

    public static readonly float StepSeconds = ParseStepSeconds();

    public static bool Enabled => StepSeconds > 0f;

    public static int Ticks => Mathf.Max(1, Mathf.RoundToInt(StepSeconds / Time.fixedDeltaTime));

    /// <summary>Let the sim advance, at most one physics tick per rendered frame.</summary>
    public static void Run()
    {
        // Just under one tick per frame, so a frame can never carry two ticks
        // and the tick count per step is exact.
        Time.captureDeltaTime = Time.fixedDeltaTime * 0.999f;
        Time.timeScale = 1f;
    }

    public static void Freeze()
    {
        Time.timeScale = 0f;
        Time.captureDeltaTime = 0f;
    }

    static float ParseStepSeconds()
    {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == "-stepSeconds"
                && float.TryParse(args[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out float s))
                return Mathf.Max(0f, s);
        }
        return DefaultStepSeconds;
    }
}
