from sd_animator import StableDiffusionAnimator, Keyframe, AnimationInterpolator

# Define your animation (change the prompts to whatever you want!)
keyframes = [
    Keyframe(
        frame=0,
        prompt="a peaceful forest in morning light, photorealistic",
        strength=0.6,
        zoom=1.0
    ),
    Keyframe(
        frame=30,
        prompt="a peaceful forest at golden hour, photorealistic",
        strength=0.65,
        zoom=1.2
    )
]

# Create the animation
interpolator = AnimationInterpolator(keyframes, total_frames=30)
animator = StableDiffusionAnimator()

# Generate frames (this takes time!)
animator.render_animation(interpolator, output_dir="my_frames", num_inference_steps=20)

# Make video
animator.create_video("my_frames", "my_animation.mp4", fps=24)