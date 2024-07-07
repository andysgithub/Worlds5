using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Model
{
    public unsafe class SurfacePoint
    {
        private float modulus; 
        private float angle;
        private float distance;

        private float xTilt;
        private float yTilt;

        // RGB colour
        public Globals.RGB_QUAD bmiColors;

        public SurfacePoint(float modulus, float angle, float distance)
        {
            this.modulus = modulus;
            this.angle = angle;
            this.distance = distance;
        }

        /// <summary>
        /// The modulus value at the surface point
        /// </summary>
        public float Modulus
        {
            get { return modulus; }
        }

        /// <summary>
        /// The angle value at the surface point
        /// </summary>
        public float Angle
        {
            get { return angle; }
        }

        /// <summary>
        /// The distance value at the surface point
        /// </summary>
        public float Distance
        {
            get { return distance; }
        }

        /// <summary>
        /// The horizontal tilt of the surface at this point (-90 to 90 degrees)
        /// </summary>
        public float XTilt
        {
            get { return xTilt; }
            set { xTilt = value; }
        }

        /// <summary>
        /// The vertical tilt of the surface at this point (-90 to 90 degrees)
        /// </summary>
        public float YTilt
        {
            get { return yTilt; }
            set { yTilt = value; }
        }
    }
}
