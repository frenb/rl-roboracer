using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TimeController : MonoBehaviour
{
    // Start is called before the first frame update
    private static float initialSpeed = 1.0f;
    public UnityEngine.UI.Slider slider;
    public UnityEngine.UI.Text sliderValue;
    public void UpdateSpeed(System.Single s)
    {
        Time.timeScale = s;
        sliderValue.text = "" + s;
    }
    void Start()
    {
        //slider.value = initialSpeed; 
        sliderValue.text = "" + initialSpeed;
        Time.timeScale = initialSpeed;
    }

    // Update is called once per frame
    void Update()
    {

    }
}
