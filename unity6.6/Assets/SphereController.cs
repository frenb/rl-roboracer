using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SphereController : MonoBehaviour
{
    // Start is called before the first frame update
    public bool animate = false;
    void Start()
    {
        if (animate)
            Destroy(gameObject, 5.0f);
    }

    // Update is called once per frame
    void Update()
    {
        // animate color change from current color to red and then destroy it
        if(animate)
            GetComponent<Renderer>().material.color = Color.Lerp(GetComponent<Renderer>().material.color, Color.red, Time.deltaTime*0.25f);
    }
}
