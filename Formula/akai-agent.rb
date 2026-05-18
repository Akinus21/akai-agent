class AkaiAgent < Formula
  desc "Remote GPU worker agent for the akai-net distributed inference system"
  homepage "https://github.com/Akinus21/akai-agent"
  version "0.0.1"
  license "MIT"

  on_linux do
    url "https://github.com/Akinus21/akai-agent/releases/download/v0.0.1/akai-agent-linux-x86_64"
    sha256 "TBD"
  end

  def install
    bin.install "akai-agent-linux-x86_64" => "akai-agent"
  end

  test do
    assert_match "akai-agent", shell_output("#{bin}/akai-agent --help")
  end
end