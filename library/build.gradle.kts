plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.compose)
    id("com.vanniktech.maven.publish") version "0.36.0"
}

val gitCommitId = providers.exec {
    commandLine("git", "rev-parse", "--short", "HEAD")
}.standardOutput.asText.map { it.trim() }.getOrElse("unknown")

android {
    namespace = "moe.grass.webgpuviewer"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("proguard-rules.txt")

        externalNativeBuild {
            cmake {
                cppFlags("-O3 -flto")
                arguments("-DANDROID_ARM_NEON=TRUE")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildFeatures {
        compose = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(libs.androidx.annotation)
    implementation(libs.androidx.webgpu)
    implementation(libs.androidx.compose.foundation)
    implementation(libs.coil.compose)
}

afterEvaluate {
    mavenPublishing {
        coordinates("moe.grass", "webgpuviewer", "1.0.1-$gitCommitId")

        pom {
            name.set("webgpuviewer")
            description.set("webgpuviewer")
            inceptionYear.set("2026")
            url.set("https://github.com/wwww-wwww/webgpuviewer/")
            licenses {
                license {
                    name.set("MIT License")
                    url.set("https://opensource.org")
                    distribution.set("repo")
                }
            }
            developers {
                developer {
                    id.set("wwww-wwww")
                    name.set("w")
                    url.set("https://github.com/wwww-wwww/")
                }
            }
            scm {
                url.set("https://github.com/wwww-wwww/webgpuviewer/")
                connection.set("scm:git:git://github.com/wwww-wwww/webgpuviewer.git")
                developerConnection.set("scm:git:ssh://git@github.com/wwww-wwww/webgpuviewer.git")
            }
        }

        publishToMavenCentral(automaticRelease = true)
        signAllPublications()
    }
}
