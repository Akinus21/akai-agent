class AkaiAgent < Formula
  desc "Remote GPU worker agent for the akai-net distributed inference system"
  homepage "https://github.com/Akinus21/akai-agent"
  version "0.0.1"
  license "MIT"

  on_linux do
    url "https://github.com/Akinus21/akai-agent/releases/download/v0.0.1/akai-agent"
    sha256 "5ffdc1248a1bf318355afdc49167659f8b50f8f4a1952d14bbd8e46afd70cff0"
  end

  def install
    bin.install "akai-agent" => "akai-agent"
  end

  test do
    assert_match "akai-agent", shell_output("#{bin}/akai-agent --help")
  end
end