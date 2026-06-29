plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "ca.mpreg.webgpuviewer.test"
    compileSdk = 37

    defaultConfig {
        applicationId = "ca.mpreg.webgpuviewer.test"
        minSdk = 24

        versionCode = 1
        versionName = "1.0.0"

        defaultConfig {
            packaging {
                jniLibs.keepDebugSymbols.addAll(listOf("*/mips/*.so", "*/mips64/*.so"))
            }
        }

        buildFeatures {
            compose = true
        }

        compileOptions {
            viewBinding.enable = true
        }
    }

    sourceSets {
        getByName("main").assets.directories.add("assets")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(project(":library"))

    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.compose.animation.core)
    implementation(libs.androidx.viewpager)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation("ca.mpreg:imagedecoder:6")
    implementation(libs.androidx.webgpu)
}
