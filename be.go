package main

import (
    "encoding/base64"
    "io/ioutil"
    "net/http"
    "os"
    "os/exec"
    "strings"

    "github.com/gin-gonic/gin"
)

type ImagePayload struct {
    ImageData string `json:"image_data"`
}

func main() {
    r := gin.Default()

    r.Use(func(c *gin.Context) {
        c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
        c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type")
        if c.Request.Method == "OPTIONS" {
            c.AbortWithStatus(204)
            return
        }
        c.Next()
    })

    r.POST("/predict", func(c *gin.Context) {
        var payload ImagePayload
        if err := c.BindJSON(&payload); err != nil {
            c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON"})
            return
        }

        commaIdx := strings.Index(payload.ImageData, ",")
        if commaIdx == -1 {
            c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid image data"})
            return
        }

        base64Str := payload.ImageData[commaIdx+1:]
        imgData, err := base64.StdEncoding.DecodeString(base64Str)
        if err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{"error": "Invalid base64 image"})
            return
        }

        // Buat folder jika belum ada
        if err := ensureDir("testAngka"); err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create directory"})
            return
        }

        imagePath := "testAngka/digit.png"
        if err := ioutil.WriteFile(imagePath, imgData, 0644); err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save image"})
            return
        }

        out, err := exec.Command("python", "predict.py", imagePath).CombinedOutput()
        if err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{
                "error":   "Prediction error",
                "details": string(out),
            })
            return
        }

        parts := strings.Split(string(out), "|")
        if len(parts) != 2 {
            c.JSON(http.StatusInternalServerError, gin.H{"error": "Invalid model response"})
            return
        }

        c.JSON(http.StatusOK, gin.H{
            "prediction": strings.TrimSpace(parts[0]),
            "confidence": strings.TrimSpace(parts[1]),
        })
    })

    r.Run(":8080")
}

// Perbaikan di sini
func ensureDir(dirName string) error {
    return os.MkdirAll(dirName, 0755)
}
