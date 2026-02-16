from suspicious_factors import detect_gan_artifacts

print("GAN artifact patterns detected:",
      detect_gan_artifacts("test.jpg"))
